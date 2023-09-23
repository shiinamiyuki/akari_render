use std::f32::consts::{FRAC_1_PI, PI};
use std::rc::Rc;

use super::BsdfEvalContext;
use super::{BsdfSample, Surface, SurfaceShader};
use crate::color::Color;
use crate::geometry::Frame;
use crate::sampling::cos_sample_hemisphere;
use crate::svm::eval::SvmEvaluator;
use crate::svm::surface::SampledWavelengths;
use crate::svm::*;

pub struct DiffuseBsdf {
    pub reflectance: Color,
}

impl Surface for DiffuseBsdf {
    fn evaluate(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        if_!(Frame::same_hemisphere(wo, wi), {
            &self.reflectance * Frame::abs_cos_theta(wi)
        }, else {
            Color::zero(ctx.color_repr)
        })
    }
    fn sample(
        &self,
        wo: Expr<Float3>,
        _u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        _swl: Var<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let wi = cos_sample_hemisphere(u_sample);
        let wi = select(Frame::same_hemisphere(wo, wi), wi, -wi);
        let pdf = Frame::abs_cos_theta(wi) * FRAC_1_PI;
        let color = &self.reflectance * Frame::abs_cos_theta(wi);
        BsdfSample {
            wi,
            pdf,
            color,
            valid: true.expr(),
        }
    }
    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        select(
            Frame::same_hemisphere(wo, wi),
            Frame::abs_cos_theta(wi) * FRAC_1_PI,
            0.0f32.expr(),
        )
    }
    fn albedo(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.reflectance * PI.expr()
    }
    fn roughness(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<color::SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        1.0f32.expr()
    }
    fn emission(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        Color::zero(ctx.color_repr)
    }
}
impl SurfaceShader for SvmDiffuseBsdf {
    fn closure(&self, svm_eval: &SvmEvaluator<'_>) -> Rc<dyn Surface> {
        let reflectance = svm_eval.eval_color(self.reflectance) * FRAC_1_PI.expr();
        Rc::new(DiffuseBsdf { reflectance })
    }
}
