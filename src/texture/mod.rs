use crate::{
    color::{Color, ColorRepr},
    interaction::{SurfaceInteraction, ShadingContext},
    *,
};

pub trait FloatTexture {
    fn evaluate(&self, si: Expr<SurfaceInteraction>, ctx: &ShadingContext<'_>) -> Expr<f32>;
}
pub trait ColorTexture {
    fn evaluate(&self, si: Expr<SurfaceInteraction>, ctx: &ShadingContext<'_>) -> Color;
}
#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct FloatTextureRef {
    pub tag: u32,
    pub index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct ColorTextureRef {
    pub tag: u32,
    pub index: u32,
}