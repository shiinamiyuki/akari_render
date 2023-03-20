use crate::{
    color::ColorRepr,
    geometry::{Frame, ShadingTriangle},
    scene::Scene,
    texture::{ColorTexture, ColorTextureRef, FloatTextureRef, FloatTexture},
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
    pub uv: Float2,
    pub dpdu: Float3,
    pub dpdv: Float3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfaceInteraction {
    pub geometry: SurfaceLocalGeometry,
    pub uv: Float2,
    pub prim_id: u32,
    pub inst_id: u32,
    pub frame: Frame,
    pub triangle: ShadingTriangle,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    LightToEye,
    EyeToLight,
}

#[derive(Clone)]
pub struct ShadingContext<'a> {
    pub scene: &'a Scene,
    pub color_repr: ColorRepr,
}
impl<'a> ShadingContext<'a> {
    pub fn color_texture(
        &self,
        tex: Expr<ColorTextureRef>,
    ) -> PolymorphicRef<'a, dyn ColorTexture> {
        self.scene.color_textures.get(tex.tag(), tex.index())
    }
    pub fn float_texture(
        &self,
        tex: Expr<FloatTextureRef>,
    ) -> PolymorphicRef<'a, dyn FloatTexture> {
        self.scene.float_textures.get(tex.tag(), tex.index())
    }
}
