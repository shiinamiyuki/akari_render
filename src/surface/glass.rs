use std::f32::consts::FRAC_1_PI;

use crate::color::Color;
use crate::geometry::Frame;
use crate::interaction::ShadingContext;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::sampling::cos_sample_hemisphere;
use crate::surface::{
    fr_dielectric, BsdfMixture, Fresnel, FresnelDielectric, MicrofacetReflection,
};
use crate::*;

use super::{Bsdf, BsdfClosure, BsdfSample, Surface};
#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct GlassSurface {
    pub color: TagIndex,
    pub roughness: f32,
    pub eta: f32,
}

impl Surface for GlassSurfaceExpr {
    fn closure(
        &self,
        si: Expr<interaction::SurfaceInteraction>,
        ctx: &ShadingContext<'_>,
    ) -> BsdfClosure {
        let color = ctx.texture(self.color());
        let color = ctx.color_from_float4(color.dispatch(|_, _, tex| tex.evaluate(si, ctx)));
        let fresnel = Box::new(FresnelDielectric {
            eta_i: const_(1.0f32),
            eta_t: self.eta(),
        });
        let reflection = Box::new(MicrofacetReflection {
            color: color.clone(),
            fresnel: fresnel.clone(),
            dist: Box::new(TrowbridgeReitzDistribution::from_roughness(make_float2(
                self.roughness(),
                self.roughness(),
            ))),
        });
        let transmission = Box::new(MicrofacetReflection {
            color,
            fresnel: fresnel.clone(),
            dist: Box::new(TrowbridgeReitzDistribution::from_roughness(make_float2(
                self.roughness(),
                self.roughness(),
            ))),
        });
        let eta = self.eta();
        let fresnel_blend = Box::new(BsdfMixture {
            frac: Box::new(move |wo, _| -> Expr<f32> {
                fr_dielectric(Frame::cos_theta(wo), const_(1.0f32), eta)
            }),
            bsdf_a: reflection,
            bsdf_b: transmission,
        });
        BsdfClosure {
            inner: fresnel_blend,
            frame: si.frame(),
        }
    }
}

impl_polymorphic!(Surface, GlassSurface);
