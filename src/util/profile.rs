use lazy_static::lazy_static;
use parking_lot::RwLock;
use std::cell::{RefCell, UnsafeCell};
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
// pub fn clear_stats() {}

static mut ENABLE: bool = false;
// static GLOBAL_STATS: RwLock<Statistics> = RwLock::new(Statistics::new());
lazy_static! {
    static ref GLOBAL_STATS: RwLock<Statistics> = RwLock::new(Statistics::new());
}
thread_local! {
    static LOCAL_STATS:RefCell<Statistics>  = RefCell::new(Statistics::new());
}
fn enabled() -> bool {
    unsafe { ENABLE }
}
pub fn print_stats() {
    flush_stats();
    let mut stats = GLOBAL_STATS.write();
    println!(
        "{:<50} {:>11} {:>13}{} {:>13}{}",
        "function", "#calls", "total", "", "avg", ""
    );

    fn sum(stats: &FuncStat) -> Duration {
        stats.total_time
            + stats
                .children
                .iter()
                .map(|(_, f)| f.total_time)
                .sum::<Duration>()
    }
    {
        stats.all.get_mut().n_calls = 1;
        stats.all.get_mut().total_time = Duration::new(0, 0);
        stats.all.get_mut().total_time = sum(stats.all.get_mut());
        print_stats_helper(&stats.all.get_mut(), 0);
    }
}
fn pretty_format_time(secs: f64) -> (f64, &'static str) {
    if secs > 1.0 {
        (secs, "s ")
    } else if secs > 1e-3 {
        (secs * 1e3, "ms")
    } else if secs > 1e-6 {
        (secs * 1e6, "us")
    } else {
        (secs * 1e9, "ns")
    }
}
fn print_stats_helper(stats: &FuncStat, level: usize) {
    let indent = " ".repeat(level);
    let name = format!("{}{}", indent, stats.name);
    let (total_time, total_unit) = pretty_format_time(stats.total_time.as_secs_f64());
    let (avg_time, avg_unit) =
        pretty_format_time(stats.total_time.as_secs_f64() / stats.n_calls as f64);
    println!(
        "{:<50} {:>11} {:>13.3}{} {:>13.3}{}",
        name, stats.n_calls, total_time, total_unit, avg_time, avg_unit
    );
    let mut sorted: Vec<_> = stats.children.keys().collect();
    sorted.sort_by_key(|i| Reverse(stats.children.get(*i).unwrap().total_time));
    for name in sorted {
        let f = stats.children.get(name).unwrap();
        print_stats_helper(f, level + 2);
    }
}
pub fn enable_profiler(enable: bool) {
    unsafe {
        ENABLE = enable;
    }
}
pub struct FuncStat {
    #[allow(dead_code)]
    pub name: &'static str,
    pub n_calls: u64,
    pub total_time: Duration,
    pub children: HashMap<&'static str, FuncStat>,
}

impl FuncStat {
    fn new(name: &'static str) -> Self {
        Self {
            n_calls: 0,
            name,
            total_time: Duration::new(0, 0),
            children: HashMap::new(),
        }
    }
    fn clear(&mut self) {
        self.n_calls = 0;
        self.total_time = Duration::new(0, 0);
        for (_, f) in &mut self.children {
            f.clear();
        }
    }
    fn with<F: FnOnce(&mut FuncStat) -> T, T>(&mut self, name: &'static str, f: F) -> T {
        loop {
            if let Some(fs) = self.children.get_mut(name) {
                return f(fs);
            } else {
                self.children.insert(name, FuncStat::new(name));
            }
        }
    }
    fn merge(&mut self, other: &FuncStat) {
        self.total_time += other.total_time;
        self.n_calls += other.n_calls;
        for (name, other_f) in &other.children {
            loop {
                if let Some(f) = self.children.get_mut(name) {
                    f.merge(other_f);
                    break;
                } else {
                    self.children.insert(name, FuncStat::new(name));
                }
            }
        }
    }
}
struct Statistics {
    all: UnsafeCell<Box<FuncStat>>,
    cur: *mut FuncStat,
}
unsafe impl Send for Statistics {}
unsafe impl Sync for Statistics {}
impl Statistics {
    fn new() -> Self {
        let mut all = Box::new(FuncStat::new("<all>"));
        let cur = all.as_mut() as *mut _;
        Self {
            all: UnsafeCell::new(all),
            cur,
        }
    }
    fn flush(&mut self) {
        let mut main = GLOBAL_STATS.write();
        unsafe {
            main.all.get_mut().merge(&*self.all.get());
            self.all.get_mut().clear();
        }
    }
}
fn flush_local_stats() {
    LOCAL_STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.flush();
    })
}
pub fn flush_stats() {
    assert!(rayon::current_thread_index().is_none());
    flush_local_stats();
    let nthr = rayon::current_num_threads();
    let flags: Vec<_> = (0..nthr).map(|_| RwLock::new(false)).collect();
    let flush = || {
        let idx = rayon::current_thread_index().unwrap();
        let mut flag = flags[idx].write();
        if *flag == false {
            flush_local_stats();
            *flag = true;
        }
    };
    use std::sync::{Condvar, Mutex};
    let done = Mutex::new(false);
    let cv = Condvar::new();
    rayon::scope(|s| {
        flush();
        loop {
            let done_ = flags.iter().all(|x| *x.read());
            if done_ {
                let mut done = done.lock().unwrap();
                *done = true;
                cv.notify_all();
                break;
            }
            s.spawn(|_| {
                flush();
                let mut done = done.lock().unwrap();
                while !*done {
                    done = cv.wait(done).unwrap();
                }
            });
        }
    });
}
impl Drop for Statistics {
    fn drop(&mut self) {
        self.flush();
    }
}

pub struct ScopedProfiler {
    #[allow(dead_code)]
    name: &'static str,
    start: Option<std::time::Instant>,
    func: *mut FuncStat,
    prev: *mut FuncStat,
}
impl ScopedProfiler {
    fn new(name: &'static str) -> Self {
        let (func, prev) = if enabled() {
            LOCAL_STATS.with(|stats| {
                let mut stats = stats.borrow_mut();
                let cur = stats.cur;
                let prev = cur;
                let cur = unsafe { &mut *cur };
                let cur = cur.with(name, |s| s as *mut FuncStat);
                stats.cur = cur;
                (cur, prev)
            })
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut())
        };
        Self {
            name,
            start: if enabled() {
                Some(std::time::Instant::now())
            } else {
                None
            },
            func,
            prev,
        }
    }
}
impl Drop for ScopedProfiler {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed();
            unsafe {
                let cur = &mut *self.func;
                cur.n_calls += 1;
                cur.total_time += elapsed;
                LOCAL_STATS.with(|stats| {
                    let mut stats = stats.borrow_mut();
                    stats.cur = self.prev;
                });
            }
        }
    }
}
#[must_use]
pub fn profile(name: &'static str) -> ScopedProfiler {
    ScopedProfiler::new(name)
}
