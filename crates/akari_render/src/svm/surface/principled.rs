use crate::color::Color;
use crate::color::*;
use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::sampling::cos_sample_hemisphere;
use crate::svm::eval::SvmEvalMode;
use crate::svm::{surface::*, SvmPrincipledBsdf};
use std::f32::consts::FRAC_1_PI;

use super::diffuse::DiffuseBsdf;
use super::{BsdfEvalContext, FresnelGeneralizedSchlick, MicrofacetTransmission, SurfaceShader};

// #[tracked(crate = "luisa")]
// fn schlick_weight(cos_theta: Expr<f32>) -> Expr<f32> {
//     let m = (1.0 - cos_theta).clamp(0.0.expr(), 1.0.expr());
//     m.sqr().sqr() * m
// }
// pub struct DisneyDiffuseBsdf {
//     pub reflectance: Color,
//     pub roughness: Expr<f32>,
// }

// impl Surface for DisneyDiffuseBsdf {
//     #[tracked(crate = "luisa")]
//     fn evaluate_impl(
//         &self,
//         wo: Expr<Float3>,
//         wi: Expr<Float3>,
//         _swl: Expr<SampledWavelengths>,
//         ctx: &BsdfEvalContext,
//     ) -> (Color, Expr<f32>) {
//         let pdf = select(
//             Frame::same_hemisphere(wo, wi),
//             Frame::abs_cos_theta(wi) * FRAC_1_PI,
//             0.0f32.expr(),
//         );
//         let color = if Frame::same_hemisphere(wo, wi) {
//             let fo = schlick_weight(Frame::abs_cos_theta(wo));
//             let fi = schlick_weight(Frame::abs_cos_theta(wi));
//             let diffuse = {
//                 // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
//                 // Burley 2015, eq (4).
//                 self.reflectance * (FRAC_1_PI * (1.0 - fo * 0.5) * (1.0 - fi * 0.5))
//             };
//             let retro = {
//                 let wh = wo + wi;
//                 let valid = wh.ne(0.0).any();
//                 let wh = wh.normalize();
//                 let cos_theta_d = wi.dot(wh);
//                 let rr = 2.0 * self.roughness * cos_theta_d * cos_theta_d;
//                 self.reflectance
//                     * select(
//                         valid,
//                         FRAC_1_PI * rr * (fo + fi + fo * fi * (rr - 1.0)),
//                         0.0f32.expr(),
//                     )
//             };
//             (diffuse + retro) * Frame::abs_cos_theta(wi)
//         } else {
//             Color::zero(ctx.color_repr)
//         };
//         (color, pdf)
//     }
//     #[tracked(crate = "luisa")]
//     fn sample_wi_impl(
//         &self,
//         wo: Expr<Float3>,
//         _u_select: Expr<f32>,
//         u_sample: Expr<Float2>,
//         _swl: Var<SampledWavelengths>,
//         _ctx: &BsdfEvalContext,
//     ) -> (Expr<Float3>, Expr<bool>) {
//         let wi = cos_sample_hemisphere(u_sample);
//         let wi = select(Frame::same_hemisphere(wo, wi), wi, -wi);
//         (wi, true.expr())
//     }
//     fn albedo_impl(
//         &self,
//         _wo: Expr<Float3>,
//         _swl: Expr<SampledWavelengths>,
//         _ctx: &BsdfEvalContext,
//     ) -> Color {
//         self.reflectance
//     }
//     fn roughness_impl(
//         &self,
//         _wo: Expr<Float3>,
//         _u_select: Expr<f32>,
//         _swl: Expr<SampledWavelengths>,
//         _ctx: &BsdfEvalContext,
//     ) -> Expr<f32> {
//         // self.roughness
//         1.0f32.expr()
//     }
//     fn emission_impl(
//         &self,
//         _wo: Expr<Float3>,
//         _swl: Expr<SampledWavelengths>,
//         ctx: &BsdfEvalContext,
//     ) -> Color {
//         Color::zero(ctx.color_repr)
//     }
// }

// pub struct DisneyFresnel {
//     pub f0: Color,
//     pub metallic: Expr<f32>,
//     pub eta: Expr<f32>,
// }
// impl Fresnel for DisneyFresnel {
//     #[tracked(crate="luisa")]
//     fn evaluate(&self, cos_theta_i: Expr<f32>, ctx: &BsdfEvalContext) -> Color {
//         let fr = Color::one(ctx.color_repr) * fr_dielectric(cos_theta_i, self.eta);
//         let f0 = fr_schlick(self.f0, cos_theta_i);
//         fr * (1.0 - self.metallic) + f0 * self.metallic
//     }
// }
impl SurfaceShader for SvmPrincipledBsdf {
    #[tracked(crate = "luisa")]
    fn closure(&self, svm_eval: &SvmEvaluator<'_>) -> Rc<dyn Surface> {
        escape!({
            if let SvmEvalMode::Alpha = svm_eval.mode() {
                let alpha = svm_eval.eval_color_alpha(self.base_color).1;
                return Rc::new(TransparentSurface {
                    inner: Rc::new(NullSurface {}),
                    alpha,
                });
            }
        });
        let (color, transmission_color, _) = {
            let (color, alpha) = svm_eval.eval_color_alpha(self.base_color);
            let transmission_color = color.sqrt();
            (color, transmission_color, alpha)
        };

        let emission = svm_eval.eval_color(self.emission_color);
        let emission = emission * svm_eval.eval_float_auto_convert(self.emission_strength);
        let metallic = svm_eval.eval_float_auto_convert(self.metallic);
        let roughness = svm_eval.eval_float_auto_convert(self.roughness);
        let eta = svm_eval.eval_float_auto_convert(self.ior);
        let transmission = svm_eval.eval_float_auto_convert(self.transmission_weight);
        // let diffuse = Rc::new(DisneyDiffuseBsdf {
        //     reflectance: color,
        //     roughness,
        // });
        let diffuse = Rc::new(DiffuseBsdf {
            reflectance: color * FRAC_1_PI.expr(),
        });
        let specular_ior_level = svm_eval.eval_float_auto_convert(self.specular_ior_level);
        let specular_tint = svm_eval.eval_color(self.specular_tint);
        let specular_tint_lum = specular_tint.lum();
        let clearcoat_weight = svm_eval.eval_float_auto_convert(self.coat_weight);
        let clearcoat_roughness = svm_eval.eval_float_auto_convert(self.coat_roughness);
        let clearcoat_ior = svm_eval.eval_float_auto_convert(self.coat_ior);
        let clearcoat_tint = svm_eval.eval_color(self.coat_tint);
        let color_repr = svm_eval.color_repr();
        let table = Rc::new(
            svm_eval
                .svm
                .get_precompute_tables(svm_eval.color_repr(), "ggx_dielectric_s"),
        );
        let (specular_weight, specular_brdf, specular_albedo) = {
            let eta = eta.var();
            let f0 = f0_from_ior(**eta).var();
            if specular_ior_level != 0.5 {
                *f0 *= 2.0 * specular_ior_level;
                *eta = ior_from_f0(**f0);
            }
            let table = table.clone();
            let specular_albedo = Rc::new(move |cos_theta_i: Expr<f32>| {
                ggx_dielectric_albedo(&table, roughness, cos_theta_i, **eta)
            });
            let fresnel = Box::new(FresnelDielectric { eta: **eta });
            // device_log!("f0: {}", **f0);
            (
                **f0,
                Rc::new(MicrofacetReflection {
                    color: specular_tint * **f0,
                    fresnel,
                    dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                        Float2::expr(roughness, roughness),
                        true,
                    )),
                }),
                specular_albedo,
            )
        };
        let (clearcoat_brdf, coat_albedo) = {
            let table = table.clone();
            let coat_albedo = Rc::new(move |cos_theta_i: Expr<f32>| {
                ggx_dielectric_albedo(&table, clearcoat_roughness, cos_theta_i, clearcoat_ior)
            });
            let fresnel = Box::new(FresnelDielectric { eta: clearcoat_ior });
            (
                Rc::new(MicrofacetReflection {
                    color: Color::one(svm_eval.color_repr()) * clearcoat_weight,
                    fresnel,
                    dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                        Float2::expr(clearcoat_roughness, clearcoat_roughness),
                        true,
                    )),
                }),
                coat_albedo,
            )
        };
        let dielectric = {
            let kr = color;
            let kt = transmission_color;
            let fresnel = Box::new(FresnelDielectric { eta });
            let roughness = svm_eval.eval_float(self.roughness);
            let reflection = Rc::new(MicrofacetReflection {
                color: kr,
                fresnel: fresnel.clone(),
                dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                    Float2::expr(roughness, roughness),
                    true,
                )),
            });
            let transmission = Rc::new(MicrofacetTransmission {
                color: kt,
                fresnel: fresnel.clone(),
                dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                    Float2::expr(roughness, roughness),
                    true,
                )),
                eta,
            });
            let dielectric = Rc::new(BsdfMixture {
                frac: Box::new(move |wo, _| -> Expr<f32> {
                    fr_dielectric(Frame::cos_theta(wo), eta)
                }),
                bsdf_a: transmission,
                bsdf_b: reflection,
                mode: BsdfBlendMode::Addictive,
            });
            dielectric
        };
        let metal = {
            let (n, k) = artistic_to_conductor_fresnel(color, specular_tint);
            Rc::new(MicrofacetReflection {
                color: Color::one(svm_eval.color_repr()),
                fresnel: Box::new(FresnelComplex { n, k }),
                // fresnel:Box::new(ConstFresnel{}),
                dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                    Float2::expr(roughness, roughness),
                    true,
                )),
            })
        };
        let bsdf = Rc::new(BsdfMixture {
            frac: Box::new(move |_, _| -> Expr<f32> { transmission }),
            bsdf_a: diffuse,
            bsdf_b: dielectric,
            mode: BsdfBlendMode::Mix,
        });

        // // mix specular and (diffuse + transmission)
        let bsdf = Rc::new(BsdfMixture {
            frac: {
                let specular_albedo = specular_albedo.clone();
                Box::new(move |wo, _| -> Expr<f32> {
                    let cos_theta = Frame::abs_cos_theta(wo);
                    let albedo = specular_albedo(cos_theta);
                    // device_log!("{} {}", albedo, specular_weight);
                    specular_weight * specular_tint_lum * albedo
                })
            },
            bsdf_a: Rc::new(ScaledBsdf {
                weight: Box::new(move |wo, ctx| -> Color {
                    let cos_theta = Frame::abs_cos_theta(wo);
                    let albedo = specular_albedo(cos_theta);
                    Color::one(ctx.color_repr) - specular_tint * specular_weight * albedo
                }),
                inner: bsdf,
            }),
            bsdf_b: specular_brdf,
            mode: BsdfBlendMode::Addictive,
        });
        // // mix metal and specular
        let bsdf = Rc::new(BsdfMixture {
            frac: Box::new(move |_, _| -> Expr<f32> { metallic }),
            bsdf_a: bsdf,
            bsdf_b: metal,
            mode: BsdfBlendMode::Mix,
        });

        // // add emission
        let bsdf = Rc::new(EmissiveSurface {
            inner: Some(bsdf),
            emission,
        });
        // mix emission and clearcoat
        let bsdf = Rc::new(BsdfMixture {
            frac: {
                let coat_albedo = coat_albedo.clone();
                Box::new(move |wo, _| -> Expr<f32> {
                    let coat_albedo = coat_albedo(Frame::abs_cos_theta(wo));
                    clearcoat_weight * coat_albedo
                })
            },
            bsdf_a: Rc::new(ScaledBsdf {
                weight: Box::new(move |wo, _| -> Color {
                    let white = Color::one(color_repr);
                    let coat_albedo = coat_albedo(Frame::abs_cos_theta(wo));
                    white.lerp_f32(clearcoat_tint * (1.0 - coat_albedo), clearcoat_weight)
                }),
                inner: bsdf,
            }),
            bsdf_b: clearcoat_brdf,
            mode: BsdfBlendMode::Addictive,
        });
        let normal = svm_eval.eval_float3_auto_convert(self.normal).var();
        *normal.x = -normal.x;
        *normal.y = -normal.y;
        let surface = Rc::new(PrincipledBsdfWrapper {
            inner: bsdf,
            albedo: color,
            emission,
        });
        normal_map(
            surface,
            **normal,
            svm_eval.si().ng,
            svm_eval.si().frame,
            NormalMapSpace::TangentSpace,
        )
    }
}

struct PrincipledBsdfWrapper {
    albedo: Color,
    emission: Color,
    inner: Rc<dyn Surface>,
}
impl Surface for PrincipledBsdfWrapper {
    fn ns(&self) -> Expr<Float3> {
        Float3::expr(0.0, 0.0, 1.0)
    }
    fn albedo_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.albedo
    }

    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        self.inner.evaluate(wo, wi, swl, ctx)
    }

    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        self.inner.sample_wi(wo, u_select, u_sample, swl, ctx)
    }

    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.inner.roughness(wo, u_select, swl, ctx)
    }

    fn emission_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.emission
    }
}
