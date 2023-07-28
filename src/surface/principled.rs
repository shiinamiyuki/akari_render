use std::f32::consts::FRAC_1_PI;

use crate::color::Color;
use crate::geometry::Frame;
use crate::interaction::SurfaceInteraction;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::sampling::cos_sample_hemisphere;
use crate::surface::*;

use super::{BsdfClosure, BsdfEvalContext, FresnelSchlick, MicrofacetTransmission, Surface};

#[derive(Debug, Clone, Copy, Value)]
#[repr(C)]
pub struct PrincipledSurface {
    pub color: TagIndex,
    pub metallic: TagIndex,
    pub roughness: TagIndex,
    pub clearcoat: TagIndex,
    pub clearcoat_roughness: TagIndex,
    pub eta: f32,
    pub transmission: TagIndex,
}

fn schlick_weight(cos_theta: Expr<f32>) -> Expr<f32> {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    m.sqr().sqr() * m
}
pub struct DisneyDiffuseBsdf {
    pub reflectance: Color,
    pub roughness: Float,
}

impl Bsdf for DisneyDiffuseBsdf {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        if_!(Frame::same_hemisphere(wo, wi), {
            let diffuse = {
                let fo = schlick_weight(Frame::abs_cos_theta(wo));
                let fi = schlick_weight(Frame::abs_cos_theta(wi));
                // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
                // Burley 2015, eq (4).
                self.reflectance * (FRAC_1_PI * (1.0 - fo * 0.5) * (1.0 - fi * 0.5))
            };
            let retro = {
                let wh = wo + wi;
                let valid = wh.cmpne(0.0).any();
                let wh = wh.normalize();
                let cos_theta_d = wi.dot(wh);
                let fo = schlick_weight(Frame::abs_cos_theta(wo));
                let fi = schlick_weight(Frame::abs_cos_theta(wi));
                let rr = 2.0 * self.roughness * cos_theta_d * cos_theta_d;
                self.reflectance
                    * select(
                        valid,
                        FRAC_1_PI * rr * (fo + fi + fo * fi * (rr - 1.0)),
                        const_(0.0f32),
                    )
            };
            (diffuse + retro) * Frame::abs_cos_theta(wi)
        }, else {
            Color::zero(ctx.color_repr)
        })
    }
    fn sample(
        &self,
        wo: Expr<Float3>,
        _u_select: Float,
        u_sample: Expr<Float2>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let wi = cos_sample_hemisphere(u_sample);
        let wi = select(
            Frame::same_hemisphere(wo, wi),
            wi,
            make_float3(wi.x(), -wi.y(), wi.x()),
        );
        let pdf = Frame::abs_cos_theta(wi) * FRAC_1_PI;
        let color = self.evaluate(wo, wi, ctx);
        BsdfSample {
            wi,
            pdf,
            color,
            valid: Bool::from(true),
            lobe_roughness: const_(1.0f32),
        }
    }
    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, _ctx: &BsdfEvalContext) -> Float {
        select(
            Frame::same_hemisphere(wo, wi),
            Frame::abs_cos_theta(wi) * FRAC_1_PI,
            Float::from(0.0),
        )
    }
    fn albedo(&self, _wo: Expr<Float3>, _ctx: &BsdfEvalContext) -> Color {
        self.reflectance
    }
}

pub struct DisneyFresnel {
    pub f0: Color,
    pub metallic: Expr<f32>,
    pub eta: Expr<f32>,
}
impl Fresnel for DisneyFresnel {
    fn evaluate(&self, cos_theta_i: Expr<f32>, ctx: &BsdfEvalContext) -> Color {
        let fr = Color::one(ctx.color_repr) * fr_dielectric(cos_theta_i, self.eta);
        let f0 = fr_schlick(self.f0, cos_theta_i);
        fr * (1.0 - self.metallic) + f0 * self.metallic
    }
}
impl Surface for PrincipledSurfaceExpr {
    fn closure(&self, si: Expr<SurfaceInteraction>, ctx: &BsdfEvalContext) -> BsdfClosure {
        let (color, transmission_color) = {
            let color = ctx.texture.evaluate_float4(self.color(), si);
            let transmission_color = ctx.texture.color_from_float4(color.sqrt());
            let color = ctx.texture.color_from_float4(color);
            (color, transmission_color)
        };
        let metallic = ctx.texture.evaluate_float(self.metallic(), si);
        let roughness = ctx.texture.evaluate_float(self.roughness(), si);
        let eta = self.eta();
        let transmission = ctx.texture.evaluate_float(self.transmission(), si);
        let diffuse = Box::new(DisneyDiffuseBsdf {
            reflectance: color,
            roughness,
        });
        let clearcoat = ctx.texture.evaluate_float(self.clearcoat(), si);
        let clearcoat_roughness = ctx.texture.evaluate_float(self.clearcoat_roughness(), si);
        let metal = {
            let f0 = ((eta - 1.0) / (eta + 1.0)).sqr();
            let f0 = Color::one(ctx.color_repr) * f0 * (1.0 - metallic) + color * metallic;
            let fresnel = Box::new(DisneyFresnel { f0, metallic, eta });
            Box::new(MicrofacetReflection {
                color,
                fresnel,
                dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                    make_float2(roughness, roughness),
                    false,
                )),
            })
        };
        let clearcoat_brdf = {
            let f0 = 0.04f32;
            let fresnel = Box::new(FresnelSchlick {
                f0: Color::one(ctx.color_repr) * const_(f0),
            });
            Box::new(MicrofacetReflection {
                color: Color::one(ctx.color_repr) * clearcoat,
                fresnel,
                dist: Box::new(TrowbridgeReitzDistribution::from_roughness(
                    make_float2(clearcoat_roughness, clearcoat_roughness),
                    false,
                )),
            })
        };
        let dieletric = {
            let kr = color;
            let kt = transmission_color;
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
            let dieletric = Box::new(BsdfMixture {
                frac: Box::new(move |wo, _| -> Expr<f32> {
                    fr_dielectric(Frame::cos_theta(wo), eta)
                }),
                bsdf_a: transmission,
                bsdf_b: reflection,
                mode: BsdfBlendMode::Mix,
            });
            dieletric
        };
        let brdf = Box::new(BsdfMixture {
            frac: Box::new(move |_, _| -> Expr<f32> { metallic }),
            bsdf_a: diffuse,
            bsdf_b: metal,
            mode: BsdfBlendMode::Mix,
        });
        let bsdf = Box::new(BsdfMixture {
            frac: Box::new(move |_, _| -> Expr<f32> { transmission }),
            bsdf_a: brdf,
            bsdf_b: dieletric,
            mode: BsdfBlendMode::Mix,
        });
        let bsdf = Box::new(BsdfMixture {
            frac: Box::new(move |_, _| -> Expr<f32> { clearcoat }),
            bsdf_a: bsdf,
            bsdf_b: clearcoat_brdf,
            mode: BsdfBlendMode::Addictive,
        });
        BsdfClosure {
            inner: bsdf,
            frame: si.frame(),
        }
    }
}

impl_polymorphic!(Surface, PrincipledSurface);
