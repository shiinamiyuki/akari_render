use crate::color::Color;
use crate::color::*;
use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::sampling::cos_sample_hemisphere;
use crate::svm::{surface::*, SvmPrincipledBsdf};
use std::f32::consts::FRAC_1_PI;

use super::{BsdfEvalContext, FresnelSchlick, MicrofacetTransmission, SurfaceShader};

#[tracked]
fn schlick_weight(cos_theta: Expr<f32>) -> Expr<f32> {
    let m = (1.0 - cos_theta).clamp(0.0.expr(), 1.0.expr());
    m.sqr().sqr() * m
}
pub struct DisneyDiffuseBsdf {
    pub reflectance: Color,
    pub roughness: Expr<f32>,
}

impl Surface for DisneyDiffuseBsdf {
    #[tracked]
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
            let fo = schlick_weight(Frame::abs_cos_theta(wo));
            let fi = schlick_weight(Frame::abs_cos_theta(wi));
            let diffuse = {
                // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
                // Burley 2015, eq (4).
                self.reflectance * (FRAC_1_PI * (1.0 - fo * 0.5) * (1.0 - fi * 0.5))
            };
            let retro = {
                let wh = wo + wi;
                let valid = wh.ne(0.0).any();
                let wh = wh.normalize();
                let cos_theta_d = wi.dot(wh);
                let rr = 2.0 * self.roughness * cos_theta_d * cos_theta_d;
                self.reflectance
                    * select(
                        valid,
                        FRAC_1_PI * rr * (fo + fi + fo * fi * (rr - 1.0)),
                        0.0f32.expr(),
                    )
            };
            (diffuse + retro) * Frame::abs_cos_theta(wi)
        } else {
            Color::zero(ctx.color_repr)
        };
        (color, pdf)
    }
    #[tracked]
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
    fn albedo_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.reflectance
    }
    fn roughness_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.roughness
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

// pub struct DisneyFresnel {
//     pub f0: Color,
//     pub metallic: Expr<f32>,
//     pub eta: Expr<f32>,
// }
// impl Fresnel for DisneyFresnel {
//     #[tracked]
//     fn evaluate(&self, cos_theta_i: Expr<f32>, ctx: &BsdfEvalContext) -> Color {
//         let fr = Color::one(ctx.color_repr) * fr_dielectric(cos_theta_i, self.eta);
//         let f0 = fr_schlick(self.f0, cos_theta_i);
//         fr * (1.0 - self.metallic) + f0 * self.metallic
//     }
// }
impl SurfaceShader for SvmPrincipledBsdf {
    #[tracked]
    fn closure(&self, svm_eval: &SvmEvaluator<'_>) -> Rc<dyn Surface> {
        let (color, transmission_color) = {
            let color = svm_eval.eval_color(self.color);
            let transmission_color = color;
            (color, transmission_color)
        };
        let lum = color.lum();
        let color_tint = select(lum.eq(0.0), color, color / lum);
        let emission = svm_eval.eval_color(self.emission);
        let metallic = svm_eval.eval_float(self.metallic);
        let roughness = svm_eval.eval_float(self.roughness);
        let eta = svm_eval.eval_float(self.eta);
        let transmission = svm_eval.eval_float(self.transmission);
        let diffuse = Rc::new(DisneyDiffuseBsdf {
            reflectance: color,
            roughness,
        });
        let specular = svm_eval.eval_float(self.specular);
        let specular_tint = svm_eval.eval_float(self.specular_tint);
        let clearcoat = svm_eval.eval_float(self.clearcoat);
        let clearcoat_roughness = svm_eval.eval_float(self.clearcoat_roughness);
        let specular_brdf = {
            let f0 = (Color::one(svm_eval.color_repr()).lerp(color_tint, specular_tint)
                * specular
                * 0.08.expr())
            .lerp(color, metallic);
            let fresnel = Box::new(FresnelSchlick { f0 });
            Rc::new(MicrofacetReflection {
                color: Color::one(svm_eval.color_repr()),
                fresnel,
                dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                    Float2::expr(roughness, roughness),
                    false,
                )),
            })
        };
        let clearcoat_brdf = {
            let f0 = 0.04f32;
            let fresnel = Box::new(FresnelSchlick {
                f0: Color::one(svm_eval.color_repr()) * f0.expr(),
            });
            Rc::new(MicrofacetReflection {
                color: Color::one(svm_eval.color_repr()) * clearcoat,
                fresnel,
                dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                    Float2::expr(clearcoat_roughness, clearcoat_roughness),
                    false,
                )),
            })
        };
        let dieletric = {
            let kr = color;
            let kt = transmission_color;
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
            let dieletric = Rc::new(BsdfMixture {
                frac: Box::new(move |wo, _| -> Expr<f32> {
                    fr_dielectric(Frame::cos_theta(wo), eta)
                }),
                bsdf_a: transmission,
                bsdf_b: reflection,
                mode: BsdfBlendMode::Mix,
            });
            dieletric
        };
        let bsdf = Rc::new(BsdfMixture {
            frac: Box::new(move |_, _| -> Expr<f32> { transmission }),
            bsdf_a: diffuse,
            bsdf_b: dieletric,
            mode: BsdfBlendMode::Mix,
        });

        let bsdf = Rc::new(BsdfMixture {
            frac: Box::new(move |_, _| -> Expr<f32> { 0.5f32.expr() }),
            bsdf_a: Rc::new(ScaledBsdf {
                inner: bsdf,
                weight: 1.0f32 - metallic,
            }),
            bsdf_b: specular_brdf,
            mode: BsdfBlendMode::Addictive,
        });
        let bsdf = Rc::new(BsdfMixture {
            frac: Box::new(move |_, _| -> Expr<f32> {
                0.0f32.expr().lerp(0.5f32.expr(), clearcoat)
            }),
            bsdf_a: bsdf,
            bsdf_b: clearcoat_brdf,
            mode: BsdfBlendMode::Addictive,
        });
        Rc::new(EmissiveSurface {
            inner: Some(bsdf),
            emission,
        })
    }
}
