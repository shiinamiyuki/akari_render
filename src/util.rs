use std::{cell::UnsafeCell, io::Read, path::PathBuf};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::*;

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
pub fn serialize_pod<T: bytemuck::Pod>(pod: T, bytes: &mut Vec<u8>) {
    bytes.extend(bytemuck::cast_slice::<T, u8>(&[pod]));
}
pub fn serialize_pods<T: bytemuck::Pod>(pods: &Vec<T>, bytes: &mut Vec<u8>) {
    serialize_pod(pods.len(), bytes);
    bytes.extend_from_slice(bytemuck::cast_slice::<T, u8>(&pods[..]));
}

pub fn deserialize_pod<T: bytemuck::Pod, In: Read>(pod: &mut T, s: &mut In) {
    let tmp = unsafe { std::slice::from_raw_parts_mut(pod as *mut T, 1) };
    s.read(bytemuck::cast_slice_mut::<T, u8>(tmp)).unwrap();
}
pub fn deserialize_pods<T: bytemuck::Pod, In: Read>(pods: &mut Vec<T>, s: &mut In) {
    let mut len: usize = 0;
    deserialize_pod(&mut len, s);
    assert!(pods.is_empty());
    pods.reserve(len);
    {
        let slice = unsafe { std::slice::from_raw_parts_mut(pods.as_mut_ptr(), len) };
        s.read(bytemuck::cast_slice_mut::<T, u8>(slice)).unwrap();
    }
}

pub fn create_progess_bar(count: usize, what: &str) -> ProgressBar {
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
    progress
}

pub struct PerThread<T> {
    data: UnsafeCell<Vec<T>>,
}
unsafe impl<T: Sync + Send + Clone> Sync for PerThread<T> {}
unsafe impl<T: Sync + Send + Clone> Send for PerThread<T> {}
impl<T: Sync + Send + Clone> PerThread<T> {
    pub fn new(v: T) -> Self {
        let num_threads = rayon::current_num_threads();
        Self {
            data: UnsafeCell::new(vec![v; num_threads]),
        }
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
pub fn erf_inv(x: Float) -> Float {
    let clamped_x: Float = clamp_t(x, -0.99999, 0.99999);
    let mut w: Float = -((1.0 as Float - clamped_x) * (1.0 as Float + clamped_x)).ln();
    let mut p: Float;
    if w < 5.0 as Float {
        w -= 2.5 as Float;
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
        w = w.sqrt() - 3.0 as Float;
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

pub fn erf(x: Float) -> Float {
    // constants
    let a1: Float = 0.254_829_592;
    let a2: Float = -0.284_496_736;
    let a3: Float = 1.421_413_741;
    let a4: Float = -1.453_152_027;
    let a5: Float = 1.061_405_429;
    let p: Float = 0.327_591_1;
    // save the sign of x
    let sign = if x < 0.0 as Float { -1.0 } else { 1.0 };
    let x: Float = x.abs();
    // A&S formula 7.1.26
    let t: Float = 1.0 as Float / (1.0 as Float + p * x);
    let y: Float =
        1.0 as Float - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

pub trait FileResolver {
    fn resolve(&self, path: &str) -> Option<std::fs::File>;
}

pub struct LocalFileResolver {
    paths: Vec<PathBuf>,
}
impl FileResolver for LocalFileResolver {
    fn resolve(&self, path: &str) -> Option<std::fs::File> {
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
