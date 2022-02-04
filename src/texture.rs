use crate::{
    shape::{Shape, SurfaceInteraction},
    util::{
        arrayvec::{ArrayVec, DynStorage, VirtualStorage},
        tiledtexture::TiledArray2D,
    },
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
    data: TiledArray2D<T, 16>,
    size: (u32, u32),
}
impl<T> ImageTexture<T>
where
    T: Copy
        + Clone
        + Sync
        + Send
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<f32, Output = T>,
{
    pub fn get_pixel_i(&self, ij: (u32, u32)) -> T {
        self.data[(ij.0 as usize, ij.1 as usize)]
    }
    pub fn get_pixel(&self, uv: Vec2) -> T {
        let mut uv = uv.fract();
        uv.y = 1.0 - uv.y;
        let i = (uv[0] * self.size.0 as f32).round() as u32 % self.size.0;
        let j = (uv[1] * self.size.1 as f32).round() as u32 % self.size.1;
        self.get_pixel_i((i, j))
    }
    pub fn width(&self) -> u32 {
        self.size.0
    }
    pub fn height(&self) -> u32 {
        self.size.1
    }
}
impl ImageTexture<Spectrum> {
    pub fn from_rgb_image(img: &image::RgbImage, use_virtual_memory: bool) -> Self {
        let data = TiledArray2D::new(
            [img.width() as usize, img.height() as usize],
            |x, y| {
                let px = img.get_pixel(x as u32, y as u32);
                let rgb = vec3(px[0] as f32, px[1] as f32, px[2] as f32) / 255.0;
                Spectrum::from_srgb(rgb)
            },
            Spectrum::zero(),
            use_virtual_memory,
        );
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
        self.get_pixel(sp.texcoord) as f32
    }
    fn power(&self) -> f32 {
        self.data.average()
    }
}
impl_base!(ImageTexture<f32>);
impl Texture for ImageTexture<Spectrum> {
    fn evaluate_s(&self, sp: &ShadingPoint) -> Spectrum {
        self.get_pixel(sp.texcoord)
    }
    fn evaluate_f(&self, sp: &ShadingPoint) -> f32 {
        self.get_pixel(sp.texcoord)[0] as f32
    }
    fn power(&self) -> f32 {
        self.data.average().samples.max_element()
    }
}
impl_base!(ImageTexture<Spectrum>);
