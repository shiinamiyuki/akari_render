use crate::{
    interaction::{ShadingContext, SurfaceInteraction},
    *,
};

pub trait Texture {
    fn evaluate(&self, si: Expr<SurfaceInteraction>, ctx: &ShadingContext<'_>) -> Expr<Float4>;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct ConstFloatTexture {
    pub value: f32,
}
impl Texture for ConstFloatTextureExpr {
    fn evaluate(&self, _si: Expr<SurfaceInteraction>, _ctx: &ShadingContext<'_>) -> Expr<Float4> {
        let v = self.value();
        make_float4(v, v, v, v)
    }
}
impl_polymorphic!(Texture, ConstFloatTexture);
#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct ConstRgbTexture {
    pub rgb: Float3,
}
impl Texture for ConstRgbTextureExpr {
    fn evaluate(&self, _si: Expr<SurfaceInteraction>, _ctx: &ShadingContext<'_>) -> Expr<Float4> {
        let rbg = self.rgb();
        make_float4(rbg.x(), rbg.y(), rbg.z(), 1.0)
    }
}
impl_polymorphic!(Texture, ConstRgbTexture);
