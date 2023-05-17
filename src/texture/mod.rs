use crate::{
    color::{Color, ColorRepr},
    interaction::SurfaceInteraction,
    scene::Scene,
    *,
};
pub struct TextureEvalContext<'a> {
    pub scene: &'a Scene,
}
pub trait Texture {
    fn evaluate(&self, si: Expr<SurfaceInteraction>, ctx: &TextureEvalContext<'_>) -> Expr<Float4>;
}

pub struct TextureEvaluator<'a> {
    pub(crate) color_repr: &'a ColorRepr,
    pub(crate) texture: Callable<(Expr<TagIndex>, Expr<SurfaceInteraction>), Expr<Float4>>,
}
impl<'a> TextureEvaluator<'a> {
    fn color_from_float4(&self, v: Expr<Float4>) -> Color {
        match self.color_repr {
            ColorRepr::Rgb => Color::Rgb(v.xyz()),
            ColorRepr::Spectral => todo!(),
        }
    }
    pub fn evaluate_color(&self, tex: Expr<TagIndex>, si: Expr<SurfaceInteraction>) -> Color {
        self.color_from_float4(self.texture.call(tex, si))
    }
    pub fn evaluate_float(&self, tex: Expr<TagIndex>, si: Expr<SurfaceInteraction>) -> Expr<f32> {
        self.texture.call(tex, si).x()
    }
}
#[repr(C)]
#[derive(Debug, Clone, Copy, Value)]
pub struct ConstFloatTexture {
    pub value: f32,
}
impl Texture for ConstFloatTextureExpr {
    fn evaluate(
        &self,
        _si: Expr<SurfaceInteraction>,
        _ctx: &TextureEvalContext<'_>,
    ) -> Expr<Float4> {
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
    fn evaluate(
        &self,
        _si: Expr<SurfaceInteraction>,
        _ctx: &TextureEvalContext<'_>,
    ) -> Expr<Float4> {
        let rbg = self.rgb();
        make_float4(rbg.x(), rbg.y(), rbg.z(), 1.0)
    }
}
impl_polymorphic!(Texture, ConstRgbTexture);
