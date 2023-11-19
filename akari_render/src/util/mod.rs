pub mod binserde;
pub mod distribution;
pub mod hash;
pub mod integration;
pub mod lerp;
pub mod profile;
use crate::{color::glam_linear_to_srgb, *};
use glam::Vec4Swizzles;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use lazy_static::lazy_static;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
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
            "[{elapsed_precise} - {eta_precise}] [{bar:50.cyan/blue}] {pos:>7}/{len:7}WHAT {msg}",
        );
        let template = template.replace("WHAT", what);
        let progress = ProgressBar::new(count as u64);
        progress.set_draw_target(ProgressDrawTarget::stdout_with_hz(2));
        progress.set_style(
            ProgressStyle::default_bar()
                .template(&template)
                .unwrap()
                .progress_chars("=>-"),
        );
        ProgressBarWrapper {
            inner: Some(progress),
        }
    } else {
        ProgressBarWrapper { inner: None }
    }
}

pub fn write_image(color: &Tex2d<Float4>, path: &str) {
    if path.ends_with(".exr") {
        write_image_hdr(color, path)
    } else {
        write_image_ldr(color, path)
    }
}
pub fn write_image_ldr(color: &Tex2d<Float4>, path: &str) {
    let color_buf = color.view(0).copy_to_vec::<Float4>();
    let parent_dir = std::path::Path::new(path).parent().unwrap();
    std::fs::create_dir_all(parent_dir).unwrap();
    let img = image::RgbImage::from_fn(color.width(), color.height(), |x, y| {
        let i = x + y * color.width();
        let pixel: glam::Vec4 = color_buf[i as usize].into();
        let rgb = pixel.xyz();
        let rgb = glam_linear_to_srgb(rgb);
        let map = |x: f32| (x * 255.0).clamp(0.0, 255.0) as u8;
        image::Rgb([map(rgb.x), map(rgb.y), map(rgb.z)])
    });
    img.save(path).unwrap();
}
pub fn write_image_hdr(color: &Tex2d<Float4>, path: &str) {
    let color_buf = color.view(0).copy_to_vec::<Float4>();
    let parent_dir = std::path::Path::new(path).parent().unwrap();
    std::fs::create_dir_all(parent_dir).unwrap();
    exr::prelude::write_rgb_file(
        path,
        color.width() as usize,
        color.height() as usize,
        |x, y| {
            let i = x + y * color.width() as usize;
            let pixel: glam::Vec4 = color_buf[i].into();
            (pixel.x, pixel.y, pixel.z)
        },
    )
    .unwrap();
}

pub fn erf_inv(x: Expr<f32>) -> Expr<f32> {
    lazy_static! {
        static ref ERF_INV: Callable<fn(Expr<f32>) -> Expr<f32>> =
            Callable::<fn(Expr<f32>) -> Expr<f32>>::new_static(track!(|x| {
                let clamped_x: Expr<f32> = x.clamp(-0.99999f32.expr(), 0.99999f32.expr());
                let w: Expr<f32> = -((1.0 - clamped_x) * (1.0 + clamped_x)).ln();
                let p = if w.lt(0.5) {
                    let w = w.var();
                    *w -= 2.5 as f32;
                    let mut p = (2.810_226_36e-08).expr();
                    p = 3.432_739_39e-07 + p * w;
                    p = -3.523_387_7e-06 + p * w;
                    p = -4.391_506_54e-06 + p * w;
                    p = 0.000_218_580_87 + p * w;
                    p = -0.001_253_725_03 + p * w;
                    p = -0.004_177_681_640 + p * w;
                    p = 0.246_640_727 + p * w;
                    p = 1.501_409_41 + p * w;
                    p
                } else {
                    let mut w = w;
                    w = w.sqrt() - 3.0 as f32;
                    let mut p = (-0.000_200_214_257).expr();
                    p = 0.000_100_950_558 + p * w;
                    p = 0.001_349_343_22 + p * w;
                    p = -0.003_673_428_44 + p * w;
                    p = 0.005_739_507_73 + p * w;
                    p = -0.007_622_461_3 + p * w;
                    p = 0.009_438_870_47 + p * w;
                    p = 1.001_674_06 + p * w;
                    p = 2.832_976_82 + p * w;
                    p
                };
                p * clamped_x
            }));
    }
    ERF_INV.call(x)
}

pub fn erf(x: Expr<f32>) -> Expr<f32> {
    lazy_static! {
        static ref ERF: Callable<fn(Expr<f32>)-> Expr<f32>> =
            Callable::<fn(Expr<f32>)-> Expr<f32>>::new_static(track!(|x| {
            // constants
            let a1: f32 = 0.254_829_592;
            let a2: f32 = -0.284_496_736;
            let a3: f32 = 1.421_413_741;
            let a4: f32 = -1.453_152_027;
            let a5: f32 = 1.061_405_429;
            let p: f32 = 0.327_591_1;
            // save the sign of x
            let sign = select(x.lt(0.0), -1.0f32.expr(), 1.0f32.expr());
            let x: Expr<f32> = x.abs();
            // A&S formula 7.1.26
            let t: Expr<f32> = 1.0 as f32 / (1.0 as f32 + p * x);
            let y: Expr<f32> =
                1.0 as f32 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
            sign * y
        }));
    }
    ERF.call(x)
}

/// Regularized lower incomplete gamma function (based on code from Cephes)
/// Based on PBRT-v4
pub fn rl_gamma(a: f64, x: f64) -> f64 {
    let eps = 1e-15f64;
    let big = 4503599627370496.0;
    let big_inv = 2.22044604925031308085e-16;

    if a < 0.0 || x < 0.0 {
        panic!("LLGamma: invalid arguments range!");
    }

    if x == 0.0 {
        return 0.0;
    }

    let ax = (a * x.ln()) - x - libm::lgamma(a);
    if ax < -709.78271289338399 {
        return if a < x { 1.0 } else { 0.0 };
    }
    if x <= 1.0 || x <= a {
        let mut r2 = a;
        let mut c2 = 1f64;
        let mut ans2 = 1f64;

        loop {
            r2 = r2 + 1.0;
            c2 = c2 * x / r2;
            ans2 += c2;
            if !((c2 / ans2) > eps) {
                break;
            }
        }

        return ax.exp() * ans2 / a;
    }

    let mut c = 0i32;
    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut p3 = 1.0;
    let mut q3 = x;
    let mut p2 = x + 1.0;
    let mut q2 = z * x;
    let mut ans = p2 / q2;

    loop {
        c += 1;
        y += 1.0;
        z += 2.0;
        let yc = y * c as f64;
        let p = (p2 * z) - (p3 * yc);
        let q = (q2 * z) - (q3 * yc);
        let error;
        if q != 0.0 {
            let nextans = p / q;
            error = ((ans - nextans) / nextans).abs();
            ans = nextans;
        } else {
            // zero div, skip
            error = 1.0;
        }

        // shift
        p3 = p2;
        p2 = p;
        q3 = q2;
        q2 = q;

        // normalize fraction when the numerator becomes large
        if p.abs() > big {
            p3 *= big_inv;
            p2 *= big_inv;
            q3 *= big_inv;
            q2 *= big_inv;
        }
        if error <= eps {
            break;
        }
    }

    return 1.0 - (ax.exp() * ans);
}

pub fn chi2cdf(x: f64, dof: i32) -> f64 {
    if dof < 1 || x < 0.0 {
        panic!("Chi2CDF: invalid arguments range!");
    } else if dof == 2 {
        1.0 - (-0.5 * x).exp()
    } else {
        rl_gamma(0.5 * dof as f64, 0.5 * x)
    }
}

pub fn mix_bits(v: Expr<u64>) -> Expr<u64> {
    lazy_static! {
        static ref MIX_BITS: Callable<fn(Expr<u64>) -> Expr<u64>> =
            Callable::<fn(Expr<u64>) -> Expr<u64>>::new_static(track!(|v: Expr<u64>| {
                let v = v.var();
                *v ^= v >> 31;
                *v *= 0x7fb5d329728ea185;
                *v ^= v >> 27;
                *v *= 0x81dadef4bc2dd44d;
                *v ^= v >> 33;
                **v
            }));
    }
    MIX_BITS.call(v)
}
#[tracked]
pub fn safe_div(a: Expr<f32>, b: Expr<f32>) -> Expr<f32> {
    select(b.eq(0.0), 0.0f32.expr(), a / b)
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new]
pub struct CompensatedSum {
    pub sum: f32,
    pub c: f32,
}
impl CompensatedSumExpr {
    #[tracked]
    pub fn update(&self, v: Expr<f32>) -> Expr<CompensatedSum> {
        let y = v - self.c;
        let t = self.sum + y;
        let c = (t - self.sum) - y;
        let sum = t;
        CompensatedSum::new_expr(sum, c)
    }
}
impl CompensatedSumVar {
    #[tracked]
    pub fn update(&self, v: Expr<f32>) {
        *self.self_ = (**self.self_).update(v);
    }
}

pub fn is_power_of_four(x: u32) -> bool {
    x != 0 && (x & (x - 1)) == 0 && (x & 0xAAAAAAAA) == 0
}
pub fn log2u32(x: u32) -> u32 {
    31 - x.leading_zeros()
}
pub fn log4u32(x: u32) -> u32 {
    log2u32(x) / 2
}
pub fn round_up_pow4(x: u32) -> u32 {
    if is_power_of_four(x) {
        x
    } else {
        1 << (log4u32(x) + 1) * 2
    }
}
pub fn round_to(x: usize, align: usize) -> usize {
    (x + align - 1) / align * align
}
#[cfg(test)]
mod test {
    fn test_log4u32_x(x: u32) {
        assert_eq!(super::log4u32(x), (x as f32).log2() as u32 / 2);
    }
    #[test]
    fn test_log4u32() {
        for i in 1..100000 {
            test_log4u32_x(i);
        }
    }
    #[test]
    fn test_round_up_pow4() {
        for i in 1..100000 {
            let x = super::round_up_pow4(i);
            assert!(super::is_power_of_four(x));
            assert!(x >= i);
            assert!(x < i * 4);
        }
    }
}

pub struct ByteVecBuilder {
    buffer: Vec<u8>,
}
impl ByteVecBuilder {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }
    pub fn push<T: Copy>(&mut self, value: T) -> usize {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let offset = round_to(self.buffer.len(), align);
        self.buffer.resize(offset + size, 0);
        unsafe {
            std::ptr::copy_nonoverlapping(
                &value as *const T,
                self.buffer.as_mut_ptr().add(offset) as *mut T,
                1,
            );
        }
        offset
    }
    pub fn finish(self) -> Vec<u8> {
        self.buffer
    }
}

pub fn with_current_dir<T>(path: &Path, f: impl FnOnce() -> T) -> T {
    let old_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(path).unwrap();
    let ret = f();
    std::env::set_current_dir(old_dir).unwrap();
    ret
}