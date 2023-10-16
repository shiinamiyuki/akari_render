use std::{
    collections::HashMap,
    fmt::{Display, Write},
    hash::Hash,
    sync::Arc,
    time::Instant,
};

use parking_lot::{Mutex, RwLock};

use crate::*;
#[derive(Clone, Copy, Debug)]
struct DispatchStat {
    total: f64,
    max: f64,
    min: f64,
    dispatch_count: u32,
}
struct DispatchToken {
    start: Option<Instant>,
}
struct DispatchProfilerInner<K: Hash + Eq + Display> {
    stats: RwLock<HashMap<K, RwLock<DispatchStat>>>,
}
pub struct DispatchProfiler<K: Hash + Eq + Display> {
    inner: Arc<DispatchProfilerInner<K>>,
}

impl<K: Hash + Eq + Display + Clone + Send + Sync + 'static> DispatchProfiler<K> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DispatchProfilerInner {
                stats: RwLock::new(HashMap::new()),
            }),
        }
    }
    /**
     * Print the profiling result to the given writer.
     * The format is:
     * | key | total | max | min | avg |
     */
    pub fn print<W: Write>(&self, f: &mut W) {
        let stats = self.inner.stats.read();
        let keys = stats.keys().collect::<Vec<_>>();
        let key_strs = keys.iter().map(|k| k.to_string()).collect::<Vec<_>>();
        let max_key_len = key_strs.iter().map(|s| s.len()).max().unwrap();
        let mut pairs = stats
            .iter()
            .map(|(k, pair)| (k.clone(), pair.read().clone()))
            .collect::<Vec<(K, DispatchStat)>>();
        pairs.sort_by(|a, b| b.1.total.partial_cmp(&a.1.total).unwrap());
        writeln!(
            f,
            "| {:width$} | {:>8} | {:>8} | {:>8} | {:>8} |",
            "name",
            "total",
            "max",
            "min",
            "avg",
            width = max_key_len
        )
        .unwrap();
        for (k, stat) in pairs {
            writeln!(
                f,
                "| {:width$} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} |",
                k,
                stat.total,
                stat.max,
                stat.min,
                stat.total / stat.dispatch_count as f64,
                width = max_key_len
            )
            .unwrap();
        }
    }
    pub fn profile<R>(&self, key: impl Into<K>, s: &Scope<'_>, f: impl FnOnce() -> R) -> R {
        let key = key.into();
        let token = Arc::new(Mutex::new(DispatchToken { start: None }));
        {
            let token = token.clone();
            s.submit_with_callback([], move || {
                let mut token = token.lock();
                token.start = Some(Instant::now());
            });
        }
        let ret = f();
        {
            let inner = self.inner.clone();
            let token = token.clone();
            s.submit_with_callback([], move || {
                let token = token.lock();
                let elapsed = token.start.unwrap().elapsed().as_secs_f64();
                {
                    let stats = inner.stats.read();
                    if !stats.contains_key(&key) {
                        drop(stats);
                        let mut stats = inner.stats.write();
                        if !stats.contains_key(&key) {
                            stats.insert(
                                key.clone(),
                                RwLock::new(DispatchStat {
                                    total: 0.0,
                                    max: 0.0,
                                    min: std::f64::INFINITY,
                                    dispatch_count: 0,
                                }),
                            );
                        }
                    }
                }
                let stats = inner.stats.read();
                let mut stat = stats.get(&key).unwrap().write();
                stat.total += elapsed;
                stat.max = stat.max.max(elapsed);
                stat.min = stat.min.min(elapsed);
                stat.dispatch_count += 1;
            });
        }
        ret
    }
}
