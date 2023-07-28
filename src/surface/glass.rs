use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::surface::{fr_dielectric, BsdfMixture, FresnelDielectric, MicrofacetReflection};
use crate::*;

use super::{BsdfBlendMode, BsdfClosure, BsdfEvalContext, MicrofacetTransmission, Surface};
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
        let fresnel = Box::new(FresnelDielectric { eta: self.eta() });
        let roughness = ctx.texture.evaluate_float(self.roughness(), si);
        let reflection = Box::new(MicrofacetReflection {
            color: kr,
            fresnel: fresnel.clone(),
            dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                make_float2(roughness, roughness),
                false,
            )),
        });
        let transmission = Box::new(MicrofacetTransmission {
            color: kt,
            fresnel: fresnel.clone(),
            dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                make_float2(roughness, roughness),
                false,
            )),
            eta: self.eta(),
        });
        let eta = self.eta();
        let fresnel_blend = Box::new(BsdfMixture {
            frac: Box::new(move |wo, _| -> Expr<f32> { fr_dielectric(Frame::cos_theta(wo), eta) }),
            bsdf_a: transmission,
            bsdf_b: reflection,
            mode: BsdfBlendMode::Mix,
        });
        BsdfClosure {
            inner: fresnel_blend,
            frame: si.frame(),
        }
    }
}

impl_polymorphic!(Surface, GlassSurface);
