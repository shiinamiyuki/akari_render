


use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::surface::{
    fr_dielectric, BsdfMixture, FresnelDielectric, MicrofacetReflection,
};
use crate::*;

use super::{BsdfClosure, Surface, MicrofacetTransmission, BsdfEvalContext};
#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct GlassSurface {
    pub kr: TagIndex,
    pub kt: TagIndex,
    pub roughness: TagIndex,
    pub eta: f32,
}

impl Surface for GlassSurfaceExpr {
    fn closure(
        &self,
        si: Expr<interaction::SurfaceInteraction>,
        ctx: &BsdfEvalContext,
    ) -> BsdfClosure {
        let kr = ctx.texture.evaluate_color(self.kr(), si);
        let kt = ctx.texture.evaluate_color(self.kt(), si);
        let fresnel = Box::new(FresnelDielectric {
            eta_i: const_(1.0f32),
            eta_t: self.eta(),
        });
        let roughness = ctx.texture.evaluate_float(self.roughness(), si);
        let reflection = Box::new(MicrofacetReflection {
            color: kr,
            fresnel: fresnel.clone(),
            dist: Box::new(TrowbridgeReitzDistribution::from_roughness(make_float2(
                roughness,
                roughness,
            ), false)),
        });
        let transmission = Box::new(MicrofacetTransmission {
            color: kt,
            fresnel: fresnel.clone(),
            dist: Box::new(TrowbridgeReitzDistribution::from_roughness(make_float2(
                roughness,
                roughness,
            ), false)),
            eta_a:const_(1.0f32),
            eta_b:self.eta(),
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
