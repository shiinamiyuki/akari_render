use std::f32::consts::{FRAC_1_PI, PI};
use std::rc::Rc;

use super::{BsdfEvalContext, NullSurface, TransparentSurface};
use super::{Surface, SurfaceShader};
use crate::color::Color;
use crate::geometry::Frame;
use crate::sampling::cos_sample_hemisphere;
use crate::svm::eval::{SvmEvalMode, SvmEvaluator};
use crate::svm::surface::SampledWavelengths;
use crate::svm::*;

pub struct DiffuseBsdf {
    pub reflectance: Color,
}

impl Surface for DiffuseBsdf {
    fn ns(&self) -> Expr<Float3> {
        Float3::expr(0.0, 0.0, 1.0)
    }
    #[tracked(crate = "luisa")]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        let pdf = select(
            Frame::same_hemisphere(wo, wi),
            Frame::abs_cos_theta(wi) * FRAC_1_PI,
            0.0f32.expr(),
        );
        let color = if Frame::same_hemisphere(wo, wi) {
            &self.reflectance * Frame::abs_cos_theta(wi)
        } else {
            Color::zero(ctx.color_repr)
        };
        (color, pdf)
    }
    #[tracked(crate = "luisa")]
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        _u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        _swl: Var<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let wi = cos_sample_hemisphere(u_sample);
        let wi = select(Frame::same_hemisphere(wo, wi), wi, -wi);
        (wi, true.expr())
    }

    #[tracked(crate = "luisa")]
    fn albedo_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.reflectance * PI.expr()
    }
    fn roughness_impl(
        &self,
        _wo: Expr<Float3>,
        _u_select: Expr<f32>,
        _swl: Expr<color::SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        1.0f32.expr()
    }
    fn emission_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        Color::zero(ctx.color_repr)
    }
}
impl SurfaceShader for SvmDiffuseBsdf {
    #[tracked(crate = "luisa")]
    fn closure(&self, svm_eval: &SvmEvaluator<'_>) -> Rc<dyn Surface> {
        let (reflectance, alpha) = svm_eval.eval_color_alpha(self.reflectance);
        escape!({
            if let SvmEvalMode::Alpha = svm_eval.mode() {
                return Rc::new(TransparentSurface {
                    inner: Rc::new(NullSurface {}),
                    alpha,
                });
            }
        });
        // Rc::new(TransparentSurface {
        //     inner: Rc::new(DiffuseBsdf {
        //         reflectance: reflectance * FRAC_1_PI.expr(),
        //     }),
        //     alpha,
        // })
        Rc::new(DiffuseBsdf {
            reflectance: reflectance * FRAC_1_PI.expr(),
        })
    }
}
