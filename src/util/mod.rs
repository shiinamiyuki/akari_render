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
            "[{elapsed_precise} - {eta_precise}] [{bar:30.cyan/blue}] {pos:>7}/{len:7}WHAT {msg}",
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
