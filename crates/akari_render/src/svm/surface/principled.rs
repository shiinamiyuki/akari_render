use crate::color::Color;
use crate::color::*;
use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::svm::eval::SvmEvalMode;
use crate::svm::{surface::*, SvmPrincipledBsdf};
use std::f32::consts::FRAC_1_PI;

use super::diffuse::DiffuseBsdf;
use super::{BsdfEvalContext, MicrofacetTransmission, SurfaceShader};
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
        let bsdf = Rc::new(CoatedBsdf {
            // frac: {
            //     let specular_albedo = specular_albedo.clone();
            //     Box::new(move |wo, _| -> Expr<f32> {
            //         let cos_theta = Frame::abs_cos_theta(wo);
            //         let albedo = specular_albedo(cos_theta);
            //         // device_log!("{} {}", albedo, specular_weight);
            //         specular_weight * specular_tint_lum * albedo
            //     })
            // },
            bottom: bsdf,
            top: specular_brdf,
            e_top: Box::new(move |wo, ctx| -> Color {
                let cos_theta = Frame::abs_cos_theta(wo);
                let albedo = specular_albedo(cos_theta);
                specular_tint * albedo * specular_weight
            }),
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
        let bsdf = Rc::new(CoatedBsdf {
            e_top: {
                let coat_albedo = coat_albedo.clone();
                Box::new(move |wo, _| -> Color {
                    let coat_albedo = coat_albedo(Frame::abs_cos_theta(wo));
                    Color::one(color_repr) * clearcoat_weight * coat_albedo
                })
            },
            bottom: Rc::new(ScaledBsdf {
                weight: Box::new(move |wo, _| -> Color {
                    let white = Color::one(color_repr);
                    white.lerp_f32(clearcoat_tint, clearcoat_weight)
                }),
                inner: bsdf,
            }),
            top: clearcoat_brdf,
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
