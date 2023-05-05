use crate::{
    color::Color,
    geometry::Frame,
    interaction::{ShadingContext, SurfaceInteraction},
    *,
};
pub mod diffuse;
#[derive(Clone, Aggregate)]
pub struct BsdfSample {
    pub wi: Expr<Float3>,
    pub pdf: Expr<f32>,
    pub color: Color, // premultiplied by cos_theta(wi) and divided by pdf
    pub valid: Bool,
}

pub trait Bsdf {
    // return f(wo, wi) * cos_theta(wi)
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &ShadingContext<'_>) -> Color;
    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Float,
        u_sample: Expr<Float2>,
        ctx: &ShadingContext<'_>,
    ) -> BsdfSample;
    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &ShadingContext<'_>) -> Float;
}

pub trait Surface {
    fn closure(&self, si: Expr<SurfaceInteraction>, ctx: &ShadingContext<'_>) -> Box<dyn Bsdf>;
}

pub struct BsdfClosure {
    inner: Box<dyn Bsdf>,
    frame: Expr<Frame>,
}

impl Bsdf for BsdfClosure {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &ShadingContext<'_>) -> Color {
        self.inner
            .evaluate(self.frame.to_local(wo), self.frame.to_local(wi), ctx)
    }

    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Float,
        u_sample: Expr<Float2>,
        ctx: &ShadingContext<'_>,
    ) -> BsdfSample {
        let sample = self
            .inner
            .sample(self.frame.to_local(wo), u_select, u_sample, ctx);
        BsdfSample {
            wi: self.frame.to_world(sample.wi),
            pdf: sample.pdf,
            color: sample.color,
            valid: sample.valid,
        }
    }

    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &ShadingContext<'_>) -> Float {
        self.inner
            .pdf(self.frame.to_local(wo), self.frame.to_local(wi), ctx)
    }
}

