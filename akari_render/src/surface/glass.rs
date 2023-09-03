use super::{BsdfBlendMode, BsdfClosure, BsdfEvalContext, MicrofacetTransmission, Surface};
use crate::color::*;
use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::surface::{fr_dielectric, BsdfMixture, FresnelDielectric, MicrofacetReflection};
use crate::svm::SvmGlassBsdfExpr;
use crate::*;

impl Surface for SvmGlassBsdfExpr {
    fn closure(&self, svm_eval: &svm::eval::SvmEvaluator<'_>) -> Box<dyn surface::Bsdf> {
        let kr = svm_eval.eval_color(self.kr());
        let kt = svm_eval.eval_color(self.kt());
        let eta = svm_eval.eval_float(self.eta());
        let fresnel = Box::new(FresnelDielectric { eta });
        let roughness = svm_eval.eval_float(self.roughness());
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
            eta,
        });
        let fresnel_blend = Box::new(BsdfMixture {
            frac: Box::new(move |wo, _| -> Expr<f32> { fr_dielectric(Frame::cos_theta(wo), eta) }),
            bsdf_a: transmission,
            bsdf_b: reflection,
            mode: BsdfBlendMode::Mix,
        });
        fresnel_blend
    }
}

