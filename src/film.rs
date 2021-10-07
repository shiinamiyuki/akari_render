use crate::*;
#[derive(Copy, Clone)]
pub struct Pixel {
    pub intensity: Spectrum,
    pub weight: Float,
}
pub struct Film {
    pub pixels: RwLock<Vec<Pixel>>,
    pub resolution: glm::UVec2,
}
impl Film {
    pub fn new(resolution: &glm::UVec2) -> Self {
        Self {
            pixels: RwLock::new(vec![
                Pixel {
                    intensity: Spectrum::zero(),
                    weight: 0.0
                };
                (resolution.x * resolution.y) as usize
            ]),
            resolution: *resolution,
        }
    }
    pub fn add_sample(&self, pixel: &glm::UVec2, value: &Spectrum, weight: Float) {
        let mut pixels = self.pixels.write().unwrap();
        let pixel = &mut (*pixels)[(pixel.x + pixel.y * self.resolution.x) as usize];
        pixel.intensity = pixel.intensity + *value;
        pixel.weight += weight;
    }
    pub fn get_pixel(&self, pixel: &glm::UVec2) -> Pixel {
        let pixels = self.pixels.read().unwrap();
        (*pixels)[(pixel.x + pixel.y * self.resolution.x) as usize]
    }
    pub fn to_rgb_image(&self) -> image::RgbImage {
        let image =
            image::ImageBuffer::from_fn(self.resolution.x, self.resolution.y, |x, y| {
                let pixel = self.get_pixel(&uvec2(x, y));
                let value = pixel.intensity * (1.0 / pixel.weight);
                let srgb = value.to_srgb() * 255.0;
                image::Rgb([srgb.x as u8, srgb.y as u8, srgb.z as u8])
            });

        image
    }
}
