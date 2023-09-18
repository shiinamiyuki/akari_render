use std::rc::Rc;

use super::{BsdfBlendMode, SurfaceClosure, BsdfEvalContext, SurfaceShader, MicrofacetTransmission};
use crate::color::*;
use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::svm::surface::{Surface, fr_dielectric, BsdfMixture, FresnelDielectric, MicrofacetReflection};
use crate::svm::SvmGlassBsdf;
use crate::*;

impl SurfaceShader for SvmGlassBsdf {
    fn closure(&self, svm_eval: &svm::eval::SvmEvaluator<'_>) -> Rc<dyn Surface> {
        let kr = svm_eval.eval_color(self.kr);
        let kt = svm_eval.eval_color(self.kt);
        let eta = svm_eval.eval_float(self.eta);
        let fresnel = Box::new(FresnelDielectric { eta });
        let roughness = svm_eval.eval_float(self.roughness);
        let reflection = Rc::new(MicrofacetReflection {
            color: kr,
            fresnel: fresnel.clone(),
            dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                Float2::expr(roughness, roughness),
                false,
            )),
        });
        let transmission = Rc::new(MicrofacetTransmission {
            color: kt,
            fresnel: fresnel.clone(),
            dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                Float2::expr(roughness, roughness),
                false,
            )),
            eta,
        });
        let fresnel_blend = Rc::new(BsdfMixture {
            frac: Box::new(move |wo, _| -> Expr<f32> { fr_dielectric(Frame::cos_theta(wo), eta) }),
            bsdf_a: transmission,
            bsdf_b: reflection,
            mode: BsdfBlendMode::Mix,
        });
        fresnel_blend
    }
}
