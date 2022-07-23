use crate::*;
use color::XYZ;
use serde::{Serialize, Deserialize};
use util::RobustSum;
#[derive(Copy, Clone, Serialize, Deserialize)]
#[repr(align(64))]
pub struct Pixel {
    pub intensity: RobustSum<XYZ>,
    pub weight: RobustSum<f32>,
    pub splat: RobustSum<XYZ>,
}
impl Pixel {
    pub fn color(&self) -> XYZ {
        let weight = self.weight.sum();
        let weight = if weight == 0.0 { 1.0 } else { weight };
        self.intensity.sum() / weight + self.splat.sum()
    }
}
pub struct Film {
    pixels: Vec<RwLock<Pixel>>,
    resolution: UVec2,
}
impl Film {
    pub fn pixels(&self) -> &[RwLock<Pixel>] {
        self.pixels.as_slice()
    }
    pub fn resolution(&self) -> UVec2 {
        self.resolution
    }
    pub fn new(resolution: &UVec2) -> Self {
        Self {
            pixels: (0..(resolution.x * resolution.y) as usize)
                .map(|_| {
                    RwLock::new(Pixel {
                        intensity: RobustSum::new(XYZ::zero()),
                        weight: RobustSum::new(0.0),
                        splat: RobustSum::new(XYZ::zero()),
                    })
                })
                .collect(),
            resolution: *resolution,
        }
    }
    pub fn add_sample(
        &self,
        pixel: UVec2,
        spectrum: SampledSpectrum,
        swl: SampledWavelengths,
        weight: f32,
    ) {
        let xyz = swl.cie_xyz(spectrum);
        self.add_sample_xyz(pixel, xyz, weight)
    }
    pub fn add_splat(&self, pixel: UVec2, spectrum: SampledSpectrum, swl: SampledWavelengths) {
        let xyz = swl.cie_xyz(spectrum);
        self.add_splat_xyz(pixel, xyz);
    }
    pub fn add_sample_xyz(&self, pixel: UVec2, value: XYZ, weight: f32) {
        let value = if value.is_black() { XYZ::zero() } else { value };
        let mut pixel = self.pixels[(pixel.x + pixel.y * self.resolution.x) as usize].write();
        pixel.intensity.add(value);
        pixel.weight.add(weight);
    }
    pub fn add_splat_xyz(&self, pixel: UVec2, value: XYZ) {
        let value = if value.is_black() { XYZ::zero() } else { value };
        let mut pixel = self.pixels[(pixel.x + pixel.y * self.resolution.x) as usize].write();
        pixel.splat.add(value);
    }
    pub fn get_pixel(&self, pixel: UVec2) -> Pixel {
        let pixel = self.pixels[(pixel.x + pixel.y * self.resolution.x) as usize].read();
        *pixel
    }
    pub fn to_rgb_image(&self) -> image::RgbImage {
        let image = image::ImageBuffer::from_fn(self.resolution.x, self.resolution.y, |x, y| {
            let pixel = self.get_pixel(uvec2(x, y));
            let value = pixel.color();
            let srgb: SRgb = value.into();
            let srgb = linear_to_srgb(srgb.values().clamp(Vec3A::ZERO, Vec3A::ONE)) * 255.0;
            // let srgb = value.to_srgb() * 255.0;
            image::Rgb([srgb.x as u8, srgb.y as u8, srgb.z as u8])
        });

        image
    }
    pub fn write_exr(&self, path: &str) {
        exr::prelude::write_rgba_file(
            path,
            self.resolution.x as usize,
            self.resolution.y as usize,
            |x, y| {
                let pixel = *self.pixels[x + y * self.resolution.x as usize].read();
                let value = pixel.color();
                let srgb: SRgb = value.into();
                (srgb.values().x, srgb.values().y, srgb.values().z, 1.0)
            },
        )
        .unwrap();
    }
    pub fn write_exr_xyz(&self, path: &str) {
        exr::prelude::write_rgba_file(
            path,
            self.resolution.x as usize,
            self.resolution.y as usize,
            |x, y| {
                let pixel = *self.pixels[x + y * self.resolution.x as usize].read();
                let value = pixel.color();
                (value.values().x, value.values().y, value.values().z, 1.0)
            },
        )
        .unwrap();
    }
}
