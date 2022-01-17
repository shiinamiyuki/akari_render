use crate::{
    arrayvec::{ArrayVec, DynStorage, VirtualStorage},
    shape::{Shape, SurfaceInteraction},
    *,
};
use nalgebra as na;
// use image
#[derive(Clone, Copy)]
pub struct ShadingPoint {
    pub texcoord: Vec2,
}
impl ShadingPoint {
    pub fn from_rayhit(shape: &dyn Shape, ray_hit: RayHit) -> Self {
        let triangle = shape.shading_triangle(ray_hit.prim_id);
        Self {
            texcoord: triangle.texcoord(ray_hit.uv),
        }
    }
}

pub trait Texture: Sync + Send + Base {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum;
    fn evaluate_f(&self, sp: &ShadingPoint) -> f32;
    fn power(&self) -> f32;
}

pub struct ConstantTexture<T: Sync + Send> {
    pub value: T,
}
impl Texture for ConstantTexture<f32> {
    fn evaluate_s(&self, _sp: &ShadingPoint) -> Spectrum {
        Spectrum {
            samples: Vec3A::from([self.value as f32; 3]),
        }
    }
    fn evaluate_f(&self, _sp: &ShadingPoint) -> f32 {
        self.value as f32
    }
    fn power(&self) -> f32 {
        self.value as f32
    }
}
impl_base!(ConstantTexture<f32>);

impl Texture for ConstantTexture<f64> {
    fn evaluate_s(&self, _sp: &ShadingPoint) -> Spectrum {
        Spectrum {
            samples: Vec3A::from([self.value as f32; 3]),
        }
    }
    fn evaluate_f(&self, _sp: &ShadingPoint) -> f32 {
        self.value as f32
    }
    fn power(&self) -> f32 {
        self.value as f32
    }
}
impl_base!(ConstantTexture<f64>);
impl Texture for ConstantTexture<Spectrum> {
    fn evaluate_s(&self, _sp: &ShadingPoint) -> Spectrum {
        self.value
    }
    fn evaluate_f(&self, _sp: &ShadingPoint) -> f32 {
        self.value[0] as f32
    }
    fn power(&self) -> f32 {
        self.value.samples.max_element() as f32
    }
}
impl_base!(ConstantTexture<Spectrum>);

pub struct ImageTexture<T: Copy + Clone + Sync + Send + 'static> {
    data: ArrayVec<T, DynStorage<T>>,
    size: (u32, u32),
}
impl<T> ImageTexture<T>
where
    T: Copy + Clone + Sync + Send + 'static,
{
    pub fn get_pixel_i(&self, ij: (u32, u32)) -> T {
        self.data[(self.size.0 * ij.1 + ij.0) as usize]
    }
    pub fn get_pixel(&self, uv: &Vec2) -> T {
        let i = (uv[0] * self.size.0 as f32).round() as u32 % self.size.0;
        let j = (uv[1] * self.size.1 as f32).round() as u32 % self.size.1;
        self.get_pixel_i((i, j))
    }
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
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
        let mut data = unsafe {
            ArrayVec::from_storage(Box::new(Vec::with_capacity(
                (img.width() * img.height()) as usize,
            )) as DynStorage<_>)
        };
        let pixels: Vec<_> = img
            .pixels()
            .map(|px| -> Spectrum {
                let rgb = vec3(px[0] as f32, px[1] as f32, px[2] as f32) / 255.0;
                Spectrum::from_srgb(rgb)
            })
            .collect();
        data.extend_from_slice(&pixels);
        Self {
            size: (img.width(), img.height()),
            data,
        }
    }
    pub fn from_rgb_image_virtual(img: &image::RgbImage) -> Self {
        let mut data = unsafe {
            ArrayVec::from_storage(Box::new(VirtualStorage::new(
                (img.width() * img.height()) as usize,
            )) as DynStorage<_>)
        };
        let pixels: Vec<_> = img
            .pixels()
            .map(|px| -> Spectrum {
                let rgb = vec3(px[0] as f32, px[1] as f32, px[2] as f32) / 255.0;
                Spectrum::from_srgb(rgb)
            })
            .collect();
        data.extend_from_slice(&pixels);
        Self {
            size: (img.width(), img.height()),
            data,
        }
    }
}

impl Texture for ImageTexture<f32> {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum {
        Spectrum {
            samples: Vec3A::from([self.evaluate_f(sp); 3]),
        }
    }
    fn evaluate_f(&self, sp: &ShadingPoint) -> f32 {
        self.get_pixel(&sp.texcoord) as f32
    }
    fn power(&self) -> f32 {
        self.data.iter().sum::<f32>() as f32 / self.data.len() as f32
    }
}
impl_base!(ImageTexture<f32>);
impl Texture for ImageTexture<f64> {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum {
        Spectrum {
            samples: Vec3A::from([self.evaluate_f(sp); 3]),
        }
    }
    fn evaluate_f(&self, sp: &ShadingPoint) -> f32 {
        self.get_pixel(&sp.texcoord) as f32
    }
    fn power(&self) -> f32 {
        self.data.iter().sum::<f64>() as f32 / self.data.len() as f32
    }
}
impl_base!(ImageTexture<f64>);
impl Texture for ImageTexture<Spectrum> {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum {
        self.get_pixel(&sp.texcoord)
    }
    fn evaluate_f(&self, sp: &ShadingPoint) -> f32 {
        self.get_pixel(&sp.texcoord)[0] as f32
    }
    fn power(&self) -> f32 {
        self.data
            .iter()
            .map(|s| s.samples.max_element())
            .sum::<f32>()
            / self.data.len() as f32
    }
}
impl_base!(ImageTexture<Spectrum>);
