pub mod alias_table;
pub mod binserde;
use crate::{color::glam_linear_to_srgb, *};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use luisa_compute::glam::Vec4Swizzles;
use std::{
    collections::HashMap,
    path::PathBuf,
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

pub fn write_image_ldr(color: &Tex2d<Float4>, path: &str) {
    let color_buf = color.view(0).copy_to_vec::<Float4>();
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

pub fn erf_inv(x: Expr<f32>) -> Expr<f32> {
    let clamped_x: Expr<f32> = x.clamp(-0.99999, 0.99999);
    let w: Expr<f32> = -((1.0 - clamped_x) * (1.0 + clamped_x)).ln();
    let p = if_!(w.cmplt(0.5), {
        let mut w = w;
        w -= 2.5 as f32;
        let mut p = Float::from(2.810_226_36e-08);
        p = 3.432_739_39e-07 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_54e-06 + p * w;
        p = 0.000_218_580_87 + p * w;
        p = -0.001_253_725_03 + p * w;
        p = -0.004_177_681_640 + p * w;
        p = 0.246_640_727 + p * w;
        p = 1.501_409_41 + p * w;
        p
    }, else {
        let mut w = w;
        w = w.sqrt() - 3.0 as f32;
        let mut p = Float::from(-0.000_200_214_257);
        p = 0.000_100_950_558 + p * w;
        p = 0.001_349_343_22 + p * w;
        p = -0.003_673_428_44 + p * w;
        p = 0.005_739_507_73 + p * w;
        p = -0.007_622_461_3 + p * w;
        p = 0.009_438_870_47 + p * w;
        p = 1.001_674_06 + p * w;
        p = 2.832_976_82 + p * w;
        p
    });
    p * clamped_x
}

pub fn erf(x: Expr<f32>) -> Expr<f32> {
    // constants
    let a1: f32 = 0.254_829_592;
    let a2: f32 = -0.284_496_736;
    let a3: f32 = 1.421_413_741;
    let a4: f32 = -1.453_152_027;
    let a5: f32 = 1.061_405_429;
    let p: f32 = 0.327_591_1;
    // save the sign of x
    let sign = select(x.cmplt(0.0), const_(-1.0f32), const_(1.0f32));
    let x: Expr<f32> = x.abs();
    // A&S formula 7.1.26
    let t: Expr<f32> = 1.0 as f32 / (1.0 as f32 + p * x);
    let y: Expr<f32> =
        1.0 as f32 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
