use std::f32::consts::{FRAC_1_PI, PI};

use super::BsdfEvalContext;
use crate::color::Color;
use crate::geometry::Frame;
use crate::sampling::cos_sample_hemisphere;
use crate::*;

use super::{Bsdf, BsdfClosure, BsdfSample, Surface};
#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct DiffuseSurface {
    pub reflectance: TagIndex,
}
pub struct DiffuseBsdf {
    pub reflectance: Color,
}

impl Bsdf for DiffuseBsdf {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
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
            lobe_roughness: const_(1.0f32),
        }
    }
    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, _ctx: &BsdfEvalContext) -> Float {
        select(
            Frame::same_hemisphere(wo, wi),
            Frame::abs_cos_theta(wi) * FRAC_1_PI,
            Float::from(0.0),
        )
    }
    fn albedo(&self, _wo: Expr<Float3>, _ctx: &BsdfEvalContext) -> Color {
        self.reflectance * const_(PI)
    }
}
impl Surface for DiffuseSurfaceExpr {
    fn closure(
        &self,
        si: Expr<interaction::SurfaceInteraction>,
        ctx: &BsdfEvalContext,
    ) -> BsdfClosure {
        let reflectance = ctx.texture.evaluate_color(self.reflectance(), si) * const_(FRAC_1_PI);
        BsdfClosure {
            inner: Box::new(DiffuseBsdf { reflectance }),
            frame: si.frame(),
        }
    }
}

impl_polymorphic!(Surface, DiffuseSurface);
