use crate::{
    varray::{VArray, VArrayMemBuilder},
    *,
};
use nalgebra as na;
// use image
#[derive(Clone, Copy)]
pub struct ShadingPoint {
    pub texcoord: Vec2,
}
impl ShadingPoint {
    pub fn from_intersection<'a>(isct: &Intersection<'a>) -> Self {
        Self {
            texcoord: isct.texcoords,
        }
    }
}
pub trait Texture: Sync + Send + Base {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum;
    fn evaluate_f(&self, sp: &ShadingPoint) -> Float;
    fn power(&self) -> Float;
}

pub struct ConstantTexture<T: Sync + Send> {
    pub value: T,
}
impl Texture for ConstantTexture<f32> {
    fn evaluate_s(&self, _sp: &ShadingPoint) -> Spectrum {
        Spectrum {
            samples: na::SVector::from_element(self.value as Float),
        }
    }
    fn evaluate_f(&self, _sp: &ShadingPoint) -> Float {
        self.value as Float
    }
    fn power(&self) -> Float {
        self.value as Float
    }
}
impl_base!(ConstantTexture<f32>);

impl Texture for ConstantTexture<f64> {
    fn evaluate_s(&self, _sp: &ShadingPoint) -> Spectrum {
        Spectrum {
            samples: na::SVector::from_element(self.value as Float),
        }
    }
    fn evaluate_f(&self, _sp: &ShadingPoint) -> Float {
        self.value as Float
    }
    fn power(&self) -> Float {
        self.value as Float
    }
}
impl_base!(ConstantTexture<f64>);
impl Texture for ConstantTexture<Spectrum> {
    fn evaluate_s(&self, _sp: &ShadingPoint) -> Spectrum {
        self.value
    }
    fn evaluate_f(&self, _sp: &ShadingPoint) -> Float {
        self.value[0] as Float
    }
    fn power(&self) -> Float {
        self.value.samples.max() as Float
    }
}
impl_base!(ConstantTexture<Spectrum>);
pub enum ImageStorage<T: Copy + Clone + Sync + Send + 'static> {
    Cached(VArray<T>),
    Direct(Vec<T>),
}
impl<T: Copy + Clone + Sync + Send + 'static> ImageStorage<T> {
    pub fn read(&self, i: usize) -> T {
        match self {
            ImageStorage::Cached(c) => c.read(i),
            ImageStorage::Direct(d) => d[i],
        }
    }
}
pub struct ImageTexture<T: Copy + Clone + Sync + Send + 'static> {
    data: ImageStorage<T>,
    size: (u32, u32),
}
impl<T> ImageTexture<T>
where
    T: Copy + Clone + Sync + Send + 'static,
{
    pub fn get_pixel_i(&self, ij: (u32, u32)) -> T {
        self.data.read((self.size.0 * ij.1 + ij.0) as usize)
    }
    pub fn get_pixel(&self, uv: &Vec2) -> T {
        let i = (uv[0] * self.size.0 as Float).round() as u32 % self.size.0;
        let j = (uv[1] * self.size.1 as Float).round() as u32 % self.size.1;
        self.get_pixel_i((i, j))
    }
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            ImageStorage::Cached(_) => panic!("use direct storage!"),
            ImageStorage::Direct(d) => d.as_slice(),
        }
    }
    pub fn width(&self) -> u32 {
        self.size.0
    }
    pub fn height(&self) -> u32 {
        self.size.1
    }
}
impl ImageTexture<Spectrum> {
    pub fn from_rgb_image(img: &image::RgbImage) -> Self {
        Self {
            size: (img.width(), img.height()),
            data: ImageStorage::Direct(
                img.pixels()
                    .map(|px| -> Spectrum {
                        let rgb = vec3(px[0] as Float, px[1] as Float, px[2] as Float) / 255.0;
                        Spectrum::from_srgb(&rgb)
                    })
                    .collect(),
            ),
        }
    }
    pub fn from_rgb_image_virtual(img: &image::RgbImage, vmem: &mut VArrayMemBuilder) -> Self {
        let pixels: Vec<_> = img
            .pixels()
            .map(|px| -> Spectrum {
                let rgb = vec3(px[0] as Float, px[1] as Float, px[2] as Float) / 255.0;
                Spectrum::from_srgb(&rgb)
            })
            .collect();
        Self {
            size: (img.width(), img.height()),
            data: ImageStorage::Cached(vmem.allocate(&pixels)),
        }
    }
}

impl Texture for ImageTexture<f32> {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum {
        Spectrum {
            samples: na::SVector::from_element(self.evaluate_f(sp)),
        }
    }
    fn evaluate_f(&self, sp: &ShadingPoint) -> Float {
        self.get_pixel(&sp.texcoord) as Float
    }
    fn power(&self) -> Float {
        match &self.data {
            ImageStorage::Direct(data) => data.iter().sum::<f32>() as Float / data.len() as Float,
            ImageStorage::Cached(data) => data.iter().sum::<f32>() as Float / data.len() as Float,
        }
    }
}
impl_base!(ImageTexture<f32>);
impl Texture for ImageTexture<f64> {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum {
        Spectrum {
            samples: na::SVector::from_element(self.evaluate_f(sp)),
        }
    }
    fn evaluate_f(&self, sp: &ShadingPoint) -> Float {
        self.get_pixel(&sp.texcoord) as Float
    }
    fn power(&self) -> Float {
        match &self.data {
            ImageStorage::Direct(data) => data.iter().sum::<f64>() as Float / data.len() as Float,
            ImageStorage::Cached(data) => data.iter().sum::<f64>() as Float / data.len() as Float,
        }
    }
}
impl_base!(ImageTexture<f64>);
impl Texture for ImageTexture<Spectrum> {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum {
        self.get_pixel(&sp.texcoord)
    }
    fn evaluate_f(&self, sp: &ShadingPoint) -> Float {
        self.get_pixel(&sp.texcoord)[0] as Float
    }
    fn power(&self) -> Float {
        match &self.data {
            ImageStorage::Direct(data) => {
                data.iter().map(|s| s.samples.max()).sum::<Float>() / data.len() as Float
            }
            ImageStorage::Cached(data) => {
                data.iter().map(|s| s.samples.max()).sum::<Float>() / data.len() as Float
            }
        }
    }
}
impl_base!(ImageTexture<Spectrum>);
