use crate::*;
#[derive(Copy, Clone)]
pub struct Pixel {
    pub intensity: Spectrum,
    pub weight: Float,
}
pub struct Film {
    pub pixels: Vec<RwLock<Pixel>>,
    pub resolution: glm::UVec2,
}
impl Film {
    pub fn new(resolution: &glm::UVec2) -> Self {
        Self {
            pixels: (0..(resolution.x * resolution.y) as usize)
                .map(|_| {
                    RwLock::new(Pixel {
                        intensity: Spectrum::zero(),
                        weight: 0.0,
                    })
                })
                .collect(),
            resolution: *resolution,
        }
    }
    pub fn add_sample(&self, pixel: &glm::UVec2, value: &Spectrum, weight: Float) {
        let value = if value.is_black() {
            Spectrum::zero()
        } else {
            *value
        };
        let mut pixel = self.pixels[(pixel.x + pixel.y * self.resolution.x) as usize].write();
        pixel.intensity = pixel.intensity + value;
        pixel.weight += weight;
    }
    pub fn get_pixel(&self, pixel: &glm::UVec2) -> Pixel {
        let pixel = self.pixels[(pixel.x + pixel.y * self.resolution.x) as usize].read();
        *pixel
    }
    pub fn to_rgb_image(&self) -> image::RgbImage {
        let image = image::ImageBuffer::from_fn(self.resolution.x, self.resolution.y, |x, y| {
            let pixel = self.get_pixel(&uvec2(x, y));
            let value = pixel.intensity * (1.0 / pixel.weight);
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
                let value = pixel.intensity * (1.0 / pixel.weight);
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
