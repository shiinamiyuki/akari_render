use crate::color::Color;
use crate::{
    color::ColorRepr,
    geometry::{Frame, ShadingTriangle},
    scene::Scene,
    texture::Texture,
    *,
};

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfacePoint {
    pub p: Float3,
    pub n: Float3,
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfaceLocalGeometry {
    pub p: Float3,
    pub ng: Float3,
    pub ns: Float3,
    // uv in UV mapping
    pub uv: Float2,
    pub dpdu: Float3,
    pub dpdv: Float3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfaceInteraction {
    pub geometry: SurfaceLocalGeometry,
    pub bary: Float2,
    pub prim_id: u32,
    pub inst_id: u32,
    pub frame: Frame,
    pub triangle: ShadingTriangle,
    pub valid: bool,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TransportDirection {
    LightToEye,
    EyeToLight,
}

#[derive(Clone)]
pub struct ShadingContext<'a> {
    pub scene: &'a Scene,
    pub color_repr: ColorRepr,
}
impl<'a> ShadingContext<'a> {
    pub fn texture(&self, tex: Expr<TagIndex>) -> PolymorphicRef<'a, PolyKey, dyn Texture> {
        self.scene.textures.get(tex)
    }
    pub fn color_from_float4(&self, v: Expr<Float4>) -> Color {
        match self.color_repr {
            ColorRepr::Rgb => Color::Rgb(v.xyz()),
            ColorRepr::Spectral(_) => todo!(),
        }
    }
}
