use std::f32::consts::{FRAC_1_PI, PI};
use std::rc::Rc;

use super::BsdfEvalContext;
use super::{Surface, BsdfSample, SurfaceShader};
use crate::color::Color;
use crate::geometry::Frame;
use crate::sampling::cos_sample_hemisphere;
use crate::svm::surface::SampledWavelengths;
use crate::svm::eval::SvmEvaluator;
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
        _u_select: Float,
        u_sample: Expr<Float2>,
        _swl: Var<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let wi = cos_sample_hemisphere(u_sample);
        let wi = select(
            Frame::same_hemisphere(wo, wi),
            wi,
            make_float3(wi.x(), -wi.y(), wi.x()),
        );
        let pdf = Frame::abs_cos_theta(wi) * FRAC_1_PI;
        let color = &self.reflectance * Frame::abs_cos_theta(wi);
        BsdfSample {
            wi,
            pdf,
            color,
            valid: Bool::from(true),
        }
    }
    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Float {
        select(
            Frame::same_hemisphere(wo, wi),
            Frame::abs_cos_theta(wi) * FRAC_1_PI,
            Float::from(0.0),
        )
    }
    fn albedo(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.reflectance * const_(PI)
    }
    fn roughness(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<color::SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        const_(1.0f32)
    }
}
impl SurfaceShader for SvmDiffuseBsdf {
    fn closure(&self, svm_eval: &SvmEvaluator<'_>) -> Rc<dyn Surface> {
        let reflectance = svm_eval.eval_color(self.reflectance) * const_(FRAC_1_PI);
        Rc::new(DiffuseBsdf { reflectance })
    }
}
