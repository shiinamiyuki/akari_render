use super::diffuse::DiffuseBsdf;
use super::{fr_dielectric_integral, SurfaceShader};
use std::rc::Rc;

use crate::color::Color;
use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::sampling::weighted_discrete_choice2_and_remap;
use crate::svm::surface::{fr_dielectric, FresnelDielectric, MicrofacetReflection, Surface};
use crate::svm::SvmPlasticBsdf;
use crate::*;
// Plastic surface from the Tungsten renderer:
// https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/RoughPlasticBsdf.cpp
//
// Original license:
//
// Copyright (c) 2014 Benedikt Bitterli <benedikt.bitterli (at) gmail (dot) com>
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from
// the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute
// it freely, subject to the following restrictions:
//
//     1. The origin of this software must not be misrepresented; you
//        must not claim that you wrote the original software. If you
//        use this software in a product, an acknowledgment in the
//        product documentation would be appreciated but is not required.
//
//     2. Altered source versions must be plainly marked as such, and
//        must not be misrepresented as being the original software.
//
//     3. This notice may not be removed or altered from any source
//        distribution.

struct PlasticBsdf {
    substrate: DiffuseBsdf,
    coat: MicrofacetReflection,
    eta: Expr<f32>,
    sigma_a: Color,
    kd_weight: Expr<f32>,
}
#[tracked(crate = "luisa")]
fn substrate_weight(fo: Expr<f32>, kd_weight: Expr<f32>) -> Expr<f32> {
    let w = kd_weight * (1.0 - fo);
    (w == 0.0).select(0.0f32.expr(), w / (w + fo))
}
impl Surface for PlasticBsdf {
    fn ns(&self) -> Expr<Float3> {
        Float3::expr(0.0, 0.0, 1.0)
    }
    #[tracked(crate = "luisa")]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<color::SampledWavelengths>,
        ctx: &super::BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        let (f_coat, pdf_coat) = self.coat.evaluate(wo, wi, swl, ctx);
        let eta = self.eta;
        let fi = fr_dielectric(Frame::abs_cos_theta(wi), eta);
        let fo = fr_dielectric(Frame::abs_cos_theta(wo), eta);
        let a = (-(self.sigma_a
            * (1.0f32 / Frame::abs_cos_theta(wi) + 1.0f32 / Frame::abs_cos_theta(wo))))
        .exp();
        let (f_substrate, pdf_substrate) = self.substrate.evaluate(wo, wi, swl, ctx);
        let f_diffuse = f_substrate * (1.0 - fi) * (1.0 - fo) * (1.0 / eta).sqr() * a;
        let substrate_weight = substrate_weight(fo, self.kd_weight);
        let f = (f_coat + f_diffuse) * Frame::abs_cos_theta(wi);
        let pdf = pdf_coat.lerp(pdf_substrate, substrate_weight);
        (f, pdf)
    }
    #[tracked(crate = "luisa")]
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<color::SampledWavelengths>,
        ctx: &super::BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let eta = self.eta;
        let fo = fr_dielectric(Frame::abs_cos_theta(wo), eta);
        let substrate_weight = substrate_weight(fo, self.kd_weight);
        let (which, remapped) = weighted_discrete_choice2_and_remap(
            substrate_weight,
            0u32.expr(),
            1u32.expr(),
            u_select,
        );
        if which == 0 {
            self.substrate.sample_wi(wo, remapped, u_sample, swl, ctx)
        } else {
            self.coat.sample_wi(wo, remapped, u_sample, swl, ctx)
        }
    }
    #[tracked(crate = "luisa")]
    fn albedo_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<color::SampledWavelengths>,
        ctx: &super::BsdfEvalContext,
    ) -> Color {
        let eta = self.eta;
        let fo = fr_dielectric(Frame::abs_cos_theta(wo), eta);
        let substrate_weight = substrate_weight(fo, self.kd_weight);
        let coat_weight = 1.0 - substrate_weight;
        let coat_albedo = self.coat.albedo(wo, swl, ctx);
        let substrate_albedo = self.substrate.albedo(wo, swl, ctx);
        coat_albedo * coat_weight + substrate_albedo * substrate_weight
    }
    #[tracked(crate = "luisa")]
    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        swl: Expr<color::SampledWavelengths>,
        ctx: &super::BsdfEvalContext,
    ) -> Expr<f32> {
        let eta = self.eta;
        let fo = fr_dielectric(Frame::abs_cos_theta(wo), eta);
        let substrate_weight = substrate_weight(fo, self.kd_weight);
        let (which, remapped) = weighted_discrete_choice2_and_remap(
            substrate_weight,
            0u32.expr(),
            1u32.expr(),
            u_select,
        );
        if which == 0 {
            self.substrate.roughness(wo, remapped, swl, ctx)
        } else {
            self.coat.roughness(wo, remapped, swl, ctx)
        }
    }
    fn emission_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<color::SampledWavelengths>,
        ctx: &super::BsdfEvalContext,
    ) -> Color {
        Color::zero(ctx.color_repr)
    }
}
impl SurfaceShader for SvmPlasticBsdf {
    #[tracked(crate = "luisa")]
    fn closure(&self, svm_eval: &svm::eval::SvmEvaluator<'_>) -> Rc<dyn Surface> {
        let kd = svm_eval.eval_color(self.kd);
        let kd_lum = kd.lum();
        let sigma_a = svm_eval.eval_color(self.sigma_a);
        let sigma_a_lum = sigma_a.lum();
        let thickness = svm_eval.eval_float(self.thickness);
        let scaled_sigma_a = sigma_a * thickness;
        let avg_transmittance = (sigma_a_lum * thickness * -2.0).exp();
        let eta = svm_eval.eval_float(self.eta);
        let diffuse_fresnel = fr_dielectric_integral(eta);
        let roughness = svm_eval.eval_float(self.roughness);
        let dist = Box::new(TrowbridgeReitzDistribution::from_roughness(
            Float2::expr(roughness, roughness),
            true,
        ));
        Rc::new(PlasticBsdf {
            substrate: DiffuseBsdf {
                reflectance: kd / (Color::one(svm_eval.color_repr()) - kd * diffuse_fresnel),
            },
            coat: MicrofacetReflection {
                color: Color::one(svm_eval.color_repr()),
                dist,
                fresnel: Box::new(FresnelDielectric { eta }),
            },
            eta,
            sigma_a: scaled_sigma_a,
            kd_weight: kd_lum * avg_transmittance,
        })
    }
}
