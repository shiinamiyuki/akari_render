use std::f32::consts::FRAC_1_PI;

use crate::geometry::Frame;
use crate::interaction::ShadingContext;
use crate::sampling::cos_sample_hemisphere;
use crate::*;
use crate::{color::Color, texture::ColorTextureRef};

use super::{Bsdf, BsdfSample, Surface};
#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct DiffuseSurface {
    pub reflectance: ColorTextureRef,
}
pub struct DiffuseBsdf {
    pub reflectance: Color,
}

impl Bsdf for DiffuseBsdf {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &ShadingContext<'_>) -> Color {
        if_!(Frame::same_hemisphere(wo, wi), {
            &self.reflectance * Float::from(FRAC_1_PI) * Frame::abs_cos_theta(wi)
        }, else {
            Color::zero(&ctx.color_repr)
        })
    }
    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, _ctx: &ShadingContext<'_>) -> Float {
        select(
            Frame::same_hemisphere(wo, wi),
            Frame::abs_cos_theta(wi) * FRAC_1_PI,
            Float::from(0.0),
        )
    }
    fn sample(
        &self,
        wo: Expr<Float3>,
        _u_select: Float,
        u_sample: Expr<Float2>,
        _ctx: &ShadingContext<'_>,
    ) -> BsdfSample {
        let wi = cos_sample_hemisphere(u_sample);
        let wi = select(
            Frame::same_hemisphere(wo, wi),
            wi,
            make_float3(wi.x(), -wi.y(), wi.x()),
        );
        let pdf = Frame::abs_cos_theta(wi) * FRAC_1_PI;
        let color = self.reflectance.clone();
        BsdfSample {
            wi,
            pdf,
            color,
            valid: Bool::from(true),
        }
    }
}
impl Surface for DiffuseSurfaceExpr {
    fn closure(
        &self,
        si: Expr<interaction::SurfaceInteraction>,
        ctx: &ShadingContext<'_>,
    ) -> Box<dyn Bsdf> {
        let reflectance = ctx.color_texture(self.reflectance());
        let reflectance = reflectance.dispatch(|tex| tex.evaluate(si, ctx));
        Box::new(DiffuseBsdf { reflectance })
    }
}

impl_polymorphic!(Surface, DiffuseSurface);
