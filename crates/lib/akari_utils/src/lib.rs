use akari_common::{
    glam::{vec3, Vec3},
    *,
};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use parking_lot::*;
use serde::{Deserialize, Serialize};
use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
};
pub mod arrayvec;
pub mod radix_sort;
#[macro_use]
pub mod nn_v2;
#[macro_use]
pub mod profile;
#[macro_use]
pub mod binserde;
pub mod cli;
pub mod fastdiv;
pub mod filecache;
pub mod image;
pub mod lrucache;
pub mod rcu;
pub mod texcache;
// #[must_use]
// pub mod vecn;
pub fn log2(mut x: u32) -> u32 {
    let mut l = 0;
    while x > 0 {
        x >>= 1;
        l += 1;
    }
    l
}
pub fn int_bits_to_float(x: i32) -> f32 {
    unsafe { std::mem::transmute(x) }
}
pub fn float_bits_to_int(x: f32) -> i32 {
    unsafe { std::mem::transmute(x) }
}

pub struct CurrentDirGuard {
    current_dir: PathBuf,
}
impl CurrentDirGuard {
    pub fn new() -> Self {
        Self {
            current_dir: std::env::current_dir().unwrap(),
        }
    }
}
impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.current_dir).unwrap();
    }
}
pub struct ProgressBarWrapper {
    inner: Option<ProgressBar>,
}

impl ProgressBarWrapper {
    pub fn inc(&self, delta: u64) {
        if let Some(pb) = &self.inner {
            pb.inc(delta);
        }
    }
    pub fn finish(&self) {
        if let Some(pb) = &self.inner {
            pb.finish();
        }
    }
}
static PB_ENABLE: AtomicBool = AtomicBool::new(true);
pub fn enable_progress_bar(enable: bool) {
    PB_ENABLE.store(enable, Ordering::Relaxed);
}
pub fn create_progess_bar(count: usize, what: &str) -> ProgressBarWrapper {
    if PB_ENABLE.load(Ordering::Relaxed) {
        let template = String::from(
            "[{elapsed_precise} - {eta_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7}WHAT {msg}",
        );
        let template = template.replace("WHAT", what);
        let progress = ProgressBar::new(count as u64);
        progress.set_draw_target(ProgressDrawTarget::stdout_with_hz(2));
        progress.set_style(
            ProgressStyle::default_bar()
                .template(&template)
                .progress_chars("=>-"),
        );
        ProgressBarWrapper {
            inner: Some(progress),
        }
    } else {
        ProgressBarWrapper { inner: None }
    }
}

pub struct PerThread<T> {
    data: UnsafeCell<Vec<T>>,
}
unsafe impl<T> Sync for PerThread<T> {}
unsafe impl<T> Send for PerThread<T> {}
impl<T> PerThread<T> {
    pub fn new<F: Fn() -> T>(f: F) -> Self {
        let num_threads = rayon::current_num_threads();
        Self {
            data: UnsafeCell::new((0..num_threads).map(|_| f()).collect()),
        }
    }
    pub fn inner(&mut self) -> &[T] {
        self.data.get_mut().as_slice()
    }
    pub fn get(&self) -> &T {
        unsafe { &self.data.get().as_ref().unwrap()[rayon::current_thread_index().unwrap()] }
    }
    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut self.data.get().as_mut().unwrap()[rayon::current_thread_index().unwrap()] }
    }
}
pub fn clamp_t<T>(val: T, low: T, high: T) -> T
where
    T: PartialOrd,
{
    let r: T;
    if val < low {
        r = low;
    } else if val > high {
        r = high;
    } else {
        r = val;
    }
    r
}

pub fn left_shift2(mut x: u64) -> u64 {
    x &= 0xffffffff;
    x = (x ^ (x << 16)) & 0x0000ffff0000ffff;
    x = (x ^ (x << 8)) & 0x00ff00ff00ff00ff;
    x = (x ^ (x << 4)) & 0x0f0f0f0f0f0f0f0f;
    x = (x ^ (x << 2)) & 0x3333333333333333;
    x = (x ^ (x << 1)) & 0x5555555555555555;
    x
}
pub fn encode_morton2(x: u64, y: u64) -> u64 {
    (left_shift2(y) << 1) | left_shift2(x)
}

pub fn erf_inv(x: f32) -> f32 {
    let clamped_x: f32 = clamp_t(x, -0.99999, 0.99999);
    let mut w: f32 = -((1.0 as f32 - clamped_x) * (1.0 as f32 + clamped_x)).ln();
    let mut p: f32;
    if w < 5.0 as f32 {
        w -= 2.5 as f32;
        p = 2.810_226_36e-08;
        p = 3.432_739_39e-07 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_54e-06 + p * w;
        p = 0.000_218_580_87 + p * w;
        p = -0.001_253_725_03 + p * w;
        p = -0.004_177_681_640 + p * w;
        p = 0.246_640_727 + p * w;
        p = 1.501_409_41 + p * w;
    } else {
        w = w.sqrt() - 3.0 as f32;
        p = -0.000_200_214_257;
        p = 0.000_100_950_558 + p * w;
        p = 0.001_349_343_22 + p * w;
        p = -0.003_673_428_44 + p * w;
        p = 0.005_739_507_73 + p * w;
        p = -0.007_622_461_3 + p * w;
        p = 0.009_438_870_47 + p * w;
        p = 1.001_674_06 + p * w;
        p = 2.832_976_82 + p * w;
    }
    p * clamped_x
}

pub fn erf(x: f32) -> f32 {
    // constants
    let a1: f32 = 0.254_829_592;
    let a2: f32 = -0.284_496_736;
    let a3: f32 = 1.421_413_741;
    let a4: f32 = -1.453_152_027;
    let a5: f32 = 1.061_405_429;
    let p: f32 = 0.327_591_1;
    // save the sign of x
    let sign = if x < 0.0 as f32 { -1.0 } else { 1.0 };
    let x: f32 = x.abs();
    // A&S formula 7.1.26
    let t: f32 = 1.0 as f32 / (1.0 as f32 + p * x);
    let y: f32 = 1.0 as f32 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

pub trait FileResolver {
    fn resolve(&self, path: &std::path::Path) -> Option<std::fs::File>;
}

pub struct LocalFileResolver {
    pub(crate) paths: Vec<PathBuf>,
}
impl LocalFileResolver {
    pub fn new(paths: Vec<PathBuf>) -> Self {
        Self { paths }
    }
}

impl FileResolver for LocalFileResolver {
    fn resolve(&self, path: &std::path::Path) -> Option<std::fs::File> {
        if let Ok(f) = std::fs::File::open(path) {
            return Some(f);
        }
        for p in &self.paths {
            if let Ok(f) = std::fs::File::open(p.join(path)) {
                return Some(f);
            }
        }
        None
    }
}

pub fn par_permute<T: Clone + Send + Sync, F: Fn(usize) -> usize + Sync + Send>(
    data: &mut [T],
    index: F,
) {
    use rayon::iter::*;
    let tmp: Vec<_> = (0..data.len())
        .into_par_iter()
        .map(|i| data[index(i)].clone())
        .collect();
    data.clone_from_slice(&tmp);
}

pub fn foreach_rayon_thread<F: Fn() + Send + Sync>(f: F) {
    assert!(rayon::current_thread_index().is_none());
    let nthr = rayon::current_num_threads();
    let flags: Vec<_> = (0..nthr).map(|_| RwLock::new(false)).collect();
    let exec = || {
        let idx = rayon::current_thread_index().unwrap();
        let mut flag = flags[idx].write();
        if *flag == false {
            f();
            *flag = true;
        }
    };
    use std::sync::{Condvar, Mutex};
    let done = Mutex::new(false);
    let cv = Condvar::new();
    rayon::scope(|s| {
        exec();
        loop {
            let done_ = flags.iter().all(|x| *x.read());
            if done_ {
                let mut done = done.lock().unwrap();
                *done = true;
                cv.notify_all();
                break;
            }
            s.spawn(|_| {
                exec();
                let mut done = done.lock().unwrap();
                while !*done {
                    done = cv.wait(done).unwrap();
                }
            });
        }
    });
}

pub fn profile_fn<F: FnOnce() -> T, T>(f: F) -> (T, f64) {
    let now = std::time::Instant::now();
    let ret = f();
    (ret, now.elapsed().as_secs_f64())
}
pub fn profile_fn_ms<F: FnOnce() -> T, T>(f: F) -> (T, f64) {
    let now = std::time::Instant::now();
    let ret = f();
    (ret, now.elapsed().as_secs_f64() * 1000.0)
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RobustSum<T> {
    sum: T,
    c: T,
}
impl<T> RobustSum<T>
where
    T: Clone + Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>,
{
    pub fn new(init: T) -> Self {
        Self {
            sum: init,
            c: init - init,
        }
    }
    pub fn add(&mut self, v: T) {
        let y = v - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }
    pub fn sum(&self) -> T {
        self.sum
    }
}
#[derive(Clone, Copy)]
pub struct UnsafePointer<T> {
    ptr: usize,
    phantom: PhantomData<T>,
}
unsafe impl<T> Sync for UnsafePointer<T> {}
unsafe impl<T> Send for UnsafePointer<T> {}
impl<T> UnsafePointer<T> {
    pub fn new(p: *mut T) -> Self {
        Self {
            ptr: p as usize,
            phantom: PhantomData {},
        }
    }
    pub fn as_ptr(&self) -> *mut T {
        self.ptr as *mut T
    }
    pub unsafe fn as_mut<'a>(&self) -> Option<&'a mut T> {
        self.as_ptr().as_mut()
    }
    pub unsafe fn as_ref<'a>(&self) -> Option<&'a T> {
        self.as_ptr().as_ref()
    }
    pub unsafe fn offset(&self, count: isize) -> Self {
        Self::new(self.as_ptr().offset(count))
    }
}

#[derive(Serialize, Deserialize)]
pub struct AtomicFloat {
    bits: AtomicU32,
}
impl Default for AtomicFloat {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl AtomicFloat {
    pub fn new(v: f32) -> Self {
        Self {
            bits: AtomicU32::new(bytemuck::cast(v)),
        }
    }
    pub fn load(&self, ordering: Ordering) -> f32 {
        bytemuck::cast(self.bits.load(ordering))
    }
    pub fn store(&self, v: f32, ordering: Ordering) {
        self.bits.store(bytemuck::cast(v), ordering)
    }
    pub fn fetch_add(&self, v: f32, ordering: Ordering) -> f32 {
        let mut oldbits = self.bits.load(ordering);
        loop {
            let newbits: u32 = bytemuck::cast(bytemuck::cast::<u32, f32>(oldbits) + v);
            match self.bits.compare_exchange_weak(
                oldbits,
                newbits,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(x) => oldbits = x,
            }
        }
        bytemuck::cast(oldbits)
    }
}
impl Clone for AtomicFloat {
    fn clone(&self) -> Self {
        Self {
            bits: AtomicU32::new(self.bits.load(Ordering::Relaxed)),
        }
    }
}

pub fn parallel_for<F: Fn(usize) -> () + Sync>(count: usize, chunk_size: usize, f: F) {
    let nthreads = rayon::current_num_threads();
    let work_counter = AtomicUsize::new(0);
    rayon::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|_| loop {
                let work = work_counter.fetch_add(chunk_size, Ordering::Relaxed);
                if work >= count {
                    return;
                }
                for i in work..(work + chunk_size).min(count) {
                    f(i);
                }
            });
        }
    });
}
#[allow(dead_code)]
pub fn parallel_for_slice<T, F: Fn(usize, &mut T) -> () + Sync>(
    slice: &mut [T],
    chunk_size: usize,
    f: F,
) {
    let p_slice = UnsafePointer::new(slice.as_mut_ptr());
    let len = slice.len();
    parallel_for(len, chunk_size, |i| {
        let slice = unsafe { std::slice::from_raw_parts_mut(p_slice.as_ptr(), len) };
        f(i, &mut slice[i]);
    });
}
#[allow(dead_code)]
pub fn parallel_for_slice_packet<T, F: Fn(usize, &mut [T]) -> () + Sync>(
    slice: &mut [T],
    chunk_size: usize,
    packet_size: usize,
    f: F,
) {
    let p_slice = UnsafePointer::new(slice.as_mut_ptr());
    let len = slice.len();
    let count = (len + packet_size - 1) / packet_size;
    parallel_for(count, chunk_size, |i| {
        let end = (i * packet_size + packet_size).min(len);
        let slice = unsafe { std::slice::from_raw_parts_mut(p_slice.as_ptr(), len) };
        f(i, &mut slice[(i * packet_size)..end]);
    });
}
#[allow(dead_code)]
pub fn parallel_for_slice2<T, U, F: Fn(usize, &mut T, &mut U) -> () + Sync>(
    slice_0: &mut [T],
    slice_1: &mut [U],
    chunk_size: usize,
    f: F,
) {
    assert_eq!(slice_0.len(), slice_1.len());
    let p_slice_0 = UnsafePointer::new(slice_0.as_mut_ptr());
    let p_slice_1 = UnsafePointer::new(slice_1.as_mut_ptr());
    let len = slice_0.len();
    parallel_for(len, chunk_size, |i| {
        let slice_0 = unsafe { std::slice::from_raw_parts_mut(p_slice_0.as_ptr(), len) };
        let slice_1 = unsafe { std::slice::from_raw_parts_mut(p_slice_1.as_ptr(), len) };
        f(i, &mut slice_0[i], &mut slice_1[i]);
    });
}
#[allow(dead_code)]
pub fn parallel_for_slice3<T, U, S, F: Fn(usize, &mut T, &mut U, &mut S) -> () + Sync>(
    slice_0: &mut [T],
    slice_1: &mut [U],
    slice_2: &mut [S],
    chunk_size: usize,
    f: F,
) {
    assert_eq!(slice_0.len(), slice_1.len());
    assert_eq!(slice_0.len(), slice_2.len());
    let p_slice_0 = UnsafePointer::new(slice_0.as_mut_ptr());
    let p_slice_1 = UnsafePointer::new(slice_1.as_mut_ptr());
    let p_slice_2 = UnsafePointer::new(slice_2.as_mut_ptr());
    let len = slice_0.len();
    parallel_for(len, chunk_size, |i| {
        let slice_0 = unsafe { std::slice::from_raw_parts_mut(p_slice_0.as_ptr(), len) };
        let slice_1 = unsafe { std::slice::from_raw_parts_mut(p_slice_1.as_ptr(), len) };
        let slice_2 = unsafe { std::slice::from_raw_parts_mut(p_slice_2.as_ptr(), len) };
        f(i, &mut slice_0[i], &mut slice_1[i], &mut slice_2[i]);
    });
}
#[allow(dead_code)]
pub fn parallel_for_slice4<
    T,
    U,
    S,
    R,
    F: Fn(usize, &mut T, &mut U, &mut S, &mut R) -> () + Sync,
>(
    slice_0: &mut [T],
    slice_1: &mut [U],
    slice_2: &mut [S],
    slice_3: &mut [R],
    chunk_size: usize,
    f: F,
) {
    assert_eq!(slice_0.len(), slice_1.len());
    assert_eq!(slice_0.len(), slice_2.len());
    let p_slice_0 = UnsafePointer::new(slice_0.as_mut_ptr());
    let p_slice_1 = UnsafePointer::new(slice_1.as_mut_ptr());
    let p_slice_2 = UnsafePointer::new(slice_2.as_mut_ptr());
    let p_slice_3 = UnsafePointer::new(slice_3.as_mut_ptr());
    let len = slice_0.len();
    parallel_for(len, chunk_size, |i| {
        let slice_0 = unsafe { std::slice::from_raw_parts_mut(p_slice_0.as_ptr(), len) };
        let slice_1 = unsafe { std::slice::from_raw_parts_mut(p_slice_1.as_ptr(), len) };
        let slice_2 = unsafe { std::slice::from_raw_parts_mut(p_slice_2.as_ptr(), len) };
        let slice_3 = unsafe { std::slice::from_raw_parts_mut(p_slice_3.as_ptr(), len) };
        f(
            i,
            &mut slice_0[i],
            &mut slice_1[i],
            &mut slice_2[i],
            &mut slice_3[i],
        );
    });
}
pub fn srgb_to_linear1(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        (((s + 0.055) / 1.055) as f32).powf(2.4)
    }
}
pub fn srgb_to_linear1_u8(s: u8) -> f32 {
    akari_const::SRGB_TO_LINEAR[s as usize]
}

pub fn srgb_to_linear_u8(rgb: [u8; 3]) -> Vec3 {
    vec3(
        srgb_to_linear1_u8(rgb[0]),
        srgb_to_linear1_u8(rgb[1]),
        srgb_to_linear1_u8(rgb[2]),
    )
}
pub fn srgb_to_linear(rgb: Vec3) -> Vec3 {
    vec3(
        srgb_to_linear1(rgb.x),
        srgb_to_linear1(rgb.y),
        srgb_to_linear1(rgb.z),
    )
}
pub fn linear_to_srgb1(l: f32) -> f32 {
    if l <= 0.0031308 {
        l * 12.92
    } else {
        l.powf(1.0 / 2.4) * 1.055 - 0.055
    }
}
pub fn linear_to_srgb(linear: Vec3) -> Vec3 {
    vec3(
        linear_to_srgb1(linear.x),
        linear_to_srgb1(linear.y),
        linear_to_srgb1(linear.z),
    )
}
pub fn rgb_to_hsv(rgb: Vec3) -> Vec3 {
    let max = rgb.max_element();
    let min = rgb.min_element();
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let h = {
        if max == min {
            0.0
        } else if max == r && g >= b {
            60.0 * (g - b) / (max - min)
        } else if max == r && g < b {
            60.0 * (g - b) / (max - min) + 360.0
        } else if max == g {
            60.0 * (b - r) / (max - min) + 120.0
        } else if max == b {
            60.0 * (r - g) / (max - min) + 240.0
        } else {
            unreachable!()
        }
    };
    let v = max;
    let s = {
        if max == 0.0 {
            0.0
        } else {
            (max - min) / max
        }
    };
    vec3(h, s, v)
}

pub fn hsv_to_rgb(hsv: Vec3) -> Vec3 {
    let h = (hsv[0] / 60.0).floor() as u32;
    let f = hsv[0] / 60.0 - h as f32;
    let p = hsv[2] * (1.0 - hsv[1]);
    let q = hsv[2] * (1.0 - f * hsv[1]);
    let t = hsv[2] * (1.0 - (1.0 - f) * hsv[1]);
    let v = hsv[2];
    let (r, g, b) = match h {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => unreachable!(),
    };
    vec3(r, g, b)
}
pub fn rgb_to_hsl(rgb: Vec3) -> Vec3 {
    let max = rgb.max_element();
    let min = rgb.min_element();
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let h = {
        if max == min {
            0.0
        } else if max == r && g >= b {
            60.0 * (g - b) / (max - min)
        } else if max == r && g < b {
            60.0 * (g - b) / (max - min) + 360.0
        } else if max == g {
            60.0 * (b - r) / (max - min) + 120.0
        } else if max == b {
            60.0 * (r - g) / (max - min) + 240.0
        } else {
            unreachable!()
        }
    };
    let l = 0.5 * (max + min);
    let s = {
        if l == 0.0 || max == min {
            0.0
        } else if 0.0 < l || l <= 0.5 {
            (max - min) / (2.0 * l)
        } else if l > 0.5 {
            (max - min) / (2.0 - 2.0 * l)
        } else {
            unreachable!()
        }
    };
    vec3(h, s, l)
}
