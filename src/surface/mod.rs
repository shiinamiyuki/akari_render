use crate::{
    color::Color,
    interaction::{ShadingContext, SurfaceInteraction},
    *,
};
#[derive(Clone, Aggregate)]
pub struct BsdfSample {
    pub wi: Expr<Float3>,
    pub pdf: Expr<f32>,
    pub color: Color,
    pub valid: Bool,
}

pub trait Bsdf {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: ShadingContext<'_>) -> Color;
    fn sample(&self, wo: Expr<Float3>, ctx: ShadingContext<'_>) -> BsdfSample;
    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: ShadingContext<'_>) -> Float;
}
pub trait Surface {
    fn closure(&self, si: Expr<SurfaceInteraction>, ctx: ShadingContext<'_>) -> Box<dyn Bsdf>;
}
