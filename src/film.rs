use crate::*;
#[derive(Copy, Clone)]
pub struct Pixel {
    pub intensity: RobustSum<Spectrum>,
    pub weight: RobustSum<f32>,
    pub splat: RobustSum<Spectrum>,
}
impl Pixel {
    pub fn radiance(&self) -> Spectrum {
        let weight = self.weight.sum();
        let weight = if weight == 0.0 { 1.0 } else { weight };
        self.intensity.sum() / weight + self.splat.sum()
    }
}
pub struct Film {
    pub pixels: Vec<RwLock<Pixel>>,
    pub resolution: UVec2,
}
impl Film {
    pub fn new(resolution: &UVec2) -> Self {
        Self {
            pixels: (0..(resolution.x * resolution.y) as usize)
                .map(|_| {
                    RwLock::new(Pixel {
                        intensity: RobustSum::new(Spectrum::zero()),
                        weight: RobustSum::new(0.0),
                        splat: RobustSum::new(Spectrum::zero()),
                    })
                })
                .collect(),
            resolution: *resolution,
        }
    }
    pub fn add_sample(&self, pixel: UVec2, value: Spectrum, weight: f32) {
        let value = if value.is_black() {
            Spectrum::zero()
        } else {
            value
        };
        let mut pixel = self.pixels[(pixel.x + pixel.y * self.resolution.x) as usize].write();
        pixel.intensity.add(value);
        pixel.weight.add(weight);
    }
    pub fn add_splat(&self, pixel: UVec2, value: Spectrum) {
        let value = if value.is_black() {
            Spectrum::zero()
        } else {
            value
        };
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
            let value = pixel.radiance();
            let srgb = value.to_srgb() * 255.0;
            image::Rgb([srgb.x as u8, srgb.y as u8, srgb.z as u8])
        });

        image
    }
    pub fn write_exr(&self, path: &str) {
        exr::prelude::write_rgba_file(
            path,
            self.resolution.x as usize,
            self.resolution.y as usize, // write an image with 2048x2048 pixels
            |x, y| {
                let pixel = *self.pixels[x + y * self.resolution.x as usize].read();
                let value = pixel.radiance();
                (
                    // generate (or lookup in your own image) an f32 rgb color for each of the 2048x2048 pixels
                    value.samples.x,
                    value.samples.y,
                    value.samples.z,
                    1.0,
                )
            },
        )
        .unwrap();
    }
}
