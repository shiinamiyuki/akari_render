use crate::{
    color::{Color, ColorRepr},
    interaction::{SurfaceInteraction, ShadingContext},
    *,
};

pub trait FloatTexture {
    fn evaluate(&self, si: Expr<SurfaceInteraction>, ctx:ShadingContext<'_>) -> Expr<f32>;
}
pub trait ColorTexture {
    fn evaluate(&self, si: Expr<SurfaceInteraction>, ctx:ShadingContext<'_>) -> Color;
}
