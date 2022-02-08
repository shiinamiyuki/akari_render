use akari_common::glam::Vec4Swizzles;
use util::RobustSum;

use crate::{
    shape::{Shape, SurfaceInteraction},
    util::image::TiledImage,
    *,
};
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

pub trait FloatTexture: Sync + Send + Base {
    fn evaluate(&self, sp: &ShadingPoint) -> f32;
    fn power(&self) -> f32;
}
pub trait SpectrumTexture: Sync + Send + Base {
    fn evaluate(&self, sp: &ShadingPoint, lambda: SampledWavelengths) -> SampledSpectrum;
    fn power(&self) -> f32;
    fn colorspace(&self) -> Option<RgbColorSpace>;
}
pub struct ConstantFloatTexture(pub f32);
impl FloatTexture for ConstantFloatTexture {
    fn evaluate(&self, _sp: &ShadingPoint) -> f32 {
        self.0 as f32
    }
    fn power(&self) -> f32 {
        self.0 as f32
    }
}
impl_base!(ConstantFloatTexture);
pub struct ConstantRgbTexture {
    rgb: Vec3,
    rep: RgbSigmoidPolynomial,
    scale: f32,
    colorspace: RgbColorSpace,
}
impl ConstantRgbTexture {
    pub fn new(mut rgb: Vec3, colorspace: RgbColorSpace) -> Self {
        let mut scale = rgb.max_element();
        if scale < 1.0 {
            scale = 1.0;
        } else {
            rgb /= scale;
        }
        let rep = colorspace.rgb2spec(rgb);
        Self {
            rep,
            rgb,
            colorspace,
            scale,
        }
    }
}
impl SpectrumTexture for ConstantRgbTexture {
    fn evaluate(&self, _sp: &ShadingPoint, lambda: SampledWavelengths) -> SampledSpectrum {
        self.rep.sample(lambda) * self.scale
    }
    fn power(&self) -> f32 {
        self.rep.max_element() * self.scale
    }
    fn colorspace(&self) -> Option<RgbColorSpace> {
        Some(self.colorspace)
    }
}
impl_base!(ConstantRgbTexture);

pub struct ImageSpectrumTexture {
    image: TiledImage,
    colorspace: RgbColorSpace,
    invert_y: bool,
}
impl ImageSpectrumTexture {
    pub fn from_rgb_image(image: &akari_common::image::RgbImage, invert_y: bool) -> Self {
        let colorspace = RgbColorSpace::new(RgbColorSpaceId::SRgb);
        Self {
            colorspace,
            image: TiledImage::from_fn(
                image.width(),
                image.height(),
                util::image::PixelFormat::SRgb8,
                |x, y| {
                    let px = image.get_pixel(x, y);
                    let rgb = vec3(px[0] as f32, px[1] as f32, px[2] as f32) / 255.0;
                    srgb_to_linear(rgb).extend(1.0)
                },
            ),
            invert_y,
        }
    }
}
impl SpectrumTexture for ImageSpectrumTexture {
    fn evaluate(&self, sp: &ShadingPoint, lambda: SampledWavelengths) -> SampledSpectrum {
        let mut tc = sp.texcoord;
        if self.invert_y {
            tc.y = 1.0 - tc.y;
        }
        let rgba = self.image.loadf(tc, util::image::WrappingMode::Repeat);
        let rep = self.colorspace.rgb2spec(rgba.xyz());
        rep.sample(lambda)
    }

    fn power(&self) -> f32 {
        let mut sum = RobustSum::new(0.0);
        for y in 0..self.image.dimension().y {
            for x in 0..self.image.dimension().x {
                let rgb = self
                    .image
                    .load(uvec2(x, y).as_ivec2(), util::image::WrappingMode::Clamp)
                    .xyz();
                let xyz = srgb_to_xyz(rgb);
                sum.add(xyz.y);
            }
        }
        sum.sum() / (self.image.dimension().x * self.image.dimension().y) as f32
    }

    fn colorspace(&self) -> Option<RgbColorSpace> {
        Some(self.colorspace)
    }
}
