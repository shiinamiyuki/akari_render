use std::sync::Arc;

use bumpalo::Bump;

use crate::texture::{FloatTexture, ShadingPoint, SpectrumTexture};
use crate::*;
pub mod ltc;
use bitflags::bitflags;
bitflags! {
    pub struct BsdfFlags : u8 {
        const NONE = 0b0;
        const DIFFUSE = 0b1;
        const GLOSSY = 0b10;
        const SPECULAR = 0b100;
        const REFLECTION = 0b1000;
        const REFRACTION = 0b10000;
        const DIFFUSE_REFLECTION = Self::DIFFUSE.bits | Self::REFLECTION.bits;
        const DIFFUSE_REFRACTION = Self::DIFFUSE.bits | Self::REFRACTION.bits;
        const GLOSSY_REFLECTION = Self::GLOSSY.bits | Self::REFLECTION.bits;
        const GLOSSY_REFRACTION = Self::GLOSSY.bits | Self::REFRACTION.bits;
        const SPECULAR_REFLECTION = Self::SPECULAR.bits | Self::REFLECTION.bits;
        const SPECULAR_REFRACTION = Self::SPECULAR.bits | Self::REFRACTION.bits;
    }
}
impl BsdfFlags {
    pub fn is_delta_only(&self) -> bool {
        self.intersects(Self::SPECULAR) && !self.intersects(Self::DIFFUSE | Self::GLOSSY)
    }
}

pub struct BsdfSample {
    pub wi: Vec3,
    pub f: SampledSpectrum,
    pub pdf: f32,
    pub flag: BsdfFlags,
}

/*
serve as a hint for some algorithms (NRC/denoising/etc..)
*/
#[derive(Clone, Copy)]
pub struct BsdfInfo {
    pub albedo: XYZ,
    pub roughness: f32,
    pub metallic: f32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportMode {
    LightToCamera, // path from light to camera, (not the other way around)
    CameraToLight,
}

pub trait Bsdf: Sync + Send + AsAny {
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        mode: TransportMode,
        lambda: &mut SampledWavelengths,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure;
    fn emission(&self) -> Option<Arc<dyn SpectrumTexture>> {
        None
    }
}
pub trait LocalBsdfClosure: Sync + Send {
    fn evaluate(&self, wo: Vec3, wi: Vec3) -> SampledSpectrum;
    fn evaluate_pdf(&self, wo: Vec3, wi: Vec3) -> f32;
    fn sample(&self, u: Vec2, wo: Vec3) -> Option<BsdfSample>;
    fn emission(&self) -> Option<&dyn SpectrumTexture> {
        None
    }
    fn flags(&self) -> BsdfFlags;
}
#[derive(Copy, Clone)]
pub struct BsdfClosure<'a> {
    pub frame: Frame,
    pub closure: &'a dyn LocalBsdfClosure,
}

impl<'a> BsdfClosure<'a> {
    pub fn evaluate(&self, wo: Vec3, wi: Vec3) -> SampledSpectrum {
        self.closure
            .evaluate(self.frame.to_local(wo), self.frame.to_local(wi))
    }
    pub fn evaluate_pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        self.closure
            .evaluate_pdf(self.frame.to_local(wo), self.frame.to_local(wi))
    }
    pub fn sample(&self, u: Vec2, wo: Vec3) -> Option<BsdfSample> {
        let mut sample = self.closure.sample(u, self.frame.to_local(wo))?;
        sample.wi = self.frame.to_world(sample.wi);
        Some(sample)
    }
    pub fn emission(&self) -> Option<Arc<dyn SpectrumTexture>> {
        None
    }
    pub fn flags(&self) -> BsdfFlags {
        self.closure.flags()
    }
}
pub struct EmissiveBsdf {
    pub base: Arc<dyn Bsdf>,
    pub emission: Arc<dyn SpectrumTexture>,
}

pub struct EmissiveBsdfClosure<'a> {
    pub base: &'a dyn LocalBsdfClosure,
    pub emission: &'a dyn SpectrumTexture,
}
impl<'a> LocalBsdfClosure for EmissiveBsdfClosure<'a> {
    fn evaluate(&self, wo: Vec3, wi: Vec3) -> SampledSpectrum {
        self.base.evaluate(wo, wi)
    }
    fn evaluate_pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        self.base.evaluate_pdf(wo, wi)
    }
    fn sample(&self, u: Vec2, wo: Vec3) -> Option<BsdfSample> {
        self.base.sample(u, wo)
    }
    fn emission(&self) -> Option<&dyn SpectrumTexture> {
        Some(self.emission)
    }
    fn flags(&self) -> BsdfFlags {
        BsdfFlags::NONE
    }
}

impl Bsdf for EmissiveBsdf {
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        mode: TransportMode,
        lambda: &mut SampledWavelengths,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        arena.alloc(EmissiveBsdfClosure {
            base: self.base.evaluate(sp, mode, lambda, arena),
            emission: self.emission.as_ref(),
        })
    }
    fn emission(&self) -> Option<Arc<dyn SpectrumTexture>> {
        Some(self.emission.clone())
    }
}
pub struct MixBsdf<A: Bsdf, B: Bsdf> {
    pub bsdf_a: A,
    pub bsdf_b: B,
    pub frac: Arc<dyn FloatTexture>,
}

// impl<A, B> Base for MixBsdf<A, B>
// where
//     A: Bsdf + 'static,
//     B: Bsdf + 'static,
// {
//     fn as_any(&self) -> &dyn Any {
//         self
//     }
//     fn as_any_mut(&mut self) -> &mut dyn Any {
//         self
//     }
//     fn type_name(&self) -> &'static str {
//         std::any::type_name::<Self>()
//     }
// }
impl<A, B> Bsdf for MixBsdf<A, B>
where
    A: Bsdf + 'static,
    B: Bsdf + 'static,
{
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        mode: TransportMode,
        lambda: &mut SampledWavelengths,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        arena.alloc(MixBsdfClosure {
            bsdf_a: self.bsdf_a.evaluate(sp, mode, lambda, arena),
            bsdf_b: self.bsdf_b.evaluate(sp, mode, lambda, arena),
            frac: self.frac.evaluate(sp),
        })
    }
}
pub struct MixBsdfClosure<'a> {
    pub bsdf_a: &'a dyn LocalBsdfClosure,
    pub bsdf_b: &'a dyn LocalBsdfClosure,
    pub frac: f32,
}
impl<'a> LocalBsdfClosure for MixBsdfClosure<'a> {
    fn flags(&self) -> BsdfFlags {
        self.bsdf_a.flags() | self.bsdf_b.flags()
    }
    fn evaluate(&self, wo: Vec3, wi: Vec3) -> SampledSpectrum {
        SampledSpectrum::lerp(
            self.bsdf_a.evaluate(wo, wi),
            self.bsdf_b.evaluate(wo, wi),
            self.frac,
        )
    }
    fn evaluate_pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        lerp(
            self.bsdf_a.evaluate_pdf(wo, wi),
            self.bsdf_b.evaluate_pdf(wo, wi),
            self.frac,
        )
    }
    fn sample(&self, u: Vec2, wo: Vec3) -> Option<BsdfSample> {
        let frac = self.frac;
        let prob = (1.0 - frac).clamp(0.0000001, 0.9999999);
        if u[0] < prob {
            let remapped_u = vec2(u[0] / prob, u[1]);
            if let Some(sample) = self.bsdf_a.sample(remapped_u, wo) {
                if sample.flag.contains(BsdfFlags::SPECULAR) {
                    Some(BsdfSample {
                        pdf: sample.pdf * prob,
                        ..sample
                    })
                } else {
                    Some(BsdfSample {
                        flag: sample.flag,
                        wi: sample.wi,
                        pdf: self.evaluate_pdf(wo, sample.wi),
                        f: self.evaluate(wo, sample.wi),
                    })
                }
            } else {
                None
            }
        } else {
            let remapped_u = vec2((u[0] - prob) / (1.0 - prob), u[1]);
            if let Some(sample) = self.bsdf_b.sample(remapped_u, wo) {
                if sample.flag.contains(BsdfFlags::SPECULAR) {
                    Some(BsdfSample {
                        pdf: sample.pdf * (1.0 - prob),
                        ..sample
                    })
                } else {
                    Some(BsdfSample {
                        flag: sample.flag,
                        wi: sample.wi,
                        pdf: self.evaluate_pdf(wo, sample.wi),
                        f: self.evaluate(wo, sample.wi),
                    })
                }
            } else {
                None
            }
        }
    }
}
// pub struct DisneyBsdf {
//     pub color: Arc<dyn Texture>,
//     pub roughness: Arc<dyn Texture>,
//     pub tint: Arc<dyn Texture>,
//     pub sheen: Arc<dyn Texture>,
// }
// struct DisneyBsdfClosure {
//     color: SampledSpectrum,
//     roughness: f32,
//     sheen: f32,
//     tint: SampledSpectrum,
//     sheen_tint: f32,
// }
// pub fn schlick_weight(cos_theta: f32) -> f32 {
//     let m = (1.0 - cos_theta).clamp(0.0, 1.0);
//     let m2 = m * m;
//     m2 * m2 * m
// }
// pub fn f_schlick_(f0: f32, cos_theta: f32) -> f32 {
//     f0 + (1.0 - f0) * (1.0 - cos_theta).powi(5)
// }
// impl DisneyBsdfClosure {
//     fn gtr1(dot_hl: f32, a: f32) -> f32 {
//         if a >= 1.0 {
//             FRAC_1_PI
//         } else {
//             let a2 = a * a;
//             (a2 - 1.0) / (PI * a2.ln() * (1.0 + (a2 - 1.0) * dot_hl * dot_hl))
//         }
//     }
//     fn separable_ggx_g1(w: Vec3, a: f32) -> f32 {
//         let a2 = a * a;
//         let cos_theta = Frame::abs_cos_theta(w);
//         1.0 / (cos_theta + (a2 + cos_theta - a2 * cos_theta * cos_theta).sqrt())
//     }
//     fn evaluate_clearcoat(clearcoat: f32, alpha: f32, wo: Vec3, wi: Vec3, wh: Vec3) -> f32 {
//         if clearcoat <= 0.0 {
//             return 0.0;
//         }
//         let abs_dot_nh = Frame::abs_cos_theta(wh);
//         let abs_dot_hl = Frame::abs_cos_theta(wi);
//         let abs_dot_hv = Frame::abs_cos_theta(wo);
//         let dot_hl = wh.dot(wi);
//         let d = Self::gtr1(abs_dot_nh, lerp_scalar(0.1, 0.001, alpha));
//         let f = f_schlick_(0.04, dot_hl);
//         let gl = Self::separable_ggx_g1(wi, 0.25);
//         let gv = Self::separable_ggx_g1(wo, 0.25);

//         0.25 * clearcoat * d * f * gl * gv
//     }
//     fn evaluate_tint(color: SampledSpectrum) -> SampledSpectrum {
//         let rgb = color.to_rgb_linear();
//         let luminance = vec3(0.3, 0.6, 0.1).dot(rgb);
//         if luminance > 0.0 {
//             color * (1.0 / luminance)
//         } else {
//             SampledSpectrum::from_rgb_linear(vec3(1.0, 1.0, 1.0))
//         }
//     }
//     fn evaluateheen(&self, sp: &ShadingPoint, wo: Vec3, wi: Vec3, wm: Vec3) -> SampledSpectrum {
//         let sheen = self.sheen;
//         if sheen <= 0.0 {
//             return SampledSpectrum::zero();
//         }
//         let dot_hl = wm.dot(wi);
//         let tint = Self::evaluate_tint(self.color);
//         SampledSpectrum::lerp(SampledSpectrum::one(), tint, self.sheen_tint) * self.sheen * schlick_weight(dot_hl)
//     }
//     fn ggx_aniso_d(wh: Vec3, ax: f32, ay: f32) -> f32 {
//         let dot_hx2 = wh.x * wh.x;
//         let dot_hy2 = wh.y * wh.y;
//         let cos2 = Frame::cos2_theta(wh);
//         let ax2 = ax * ax;
//         let ay2 = ay * ay;
//         1.0 / (PI * ax * ay * (dot_hx2 / ax2 + dot_hy2 / ay2 + cos2).powi(2))
//     }
//     fn separable_smith_ggx_g1(w: Vec3, wh: Vec3, ax: f32, ay: f32) -> f32 {
//         let dot_hw = wh.dot(w);
//         if dot_hw <= 0.0 {
//             return 0.0;
//         }
//         let abs_tan_theta = Frame::tan_theta(w).abs();
//         if abs_tan_theta.is_infinite() {
//             return 0.0;
//         }
//         let a = (Frame::cos2_phi(w) * ax * ax + Frame::sin2_phi(w) * ay * ay).sqrt();
//         let a2_tan = (a * abs_tan_theta).powi(2);
//         let lambda = 0.5 * (-1.0 + (1.0 + a2_tan).sqrt());
//         1.0 / (1.0 + lambda)
//     }
// }

pub struct DiffuseBsdf {
    pub color: Arc<dyn SpectrumTexture>,
}

impl Bsdf for DiffuseBsdf {
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        _mode: TransportMode,
        lambda: &mut SampledWavelengths,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        arena.alloc(DiffuseBsdfClosure {
            color: self.color.evaluate(sp, lambda),
        })
    }
}
pub struct DiffuseBsdfClosure {
    pub color: SampledSpectrum,
}
impl LocalBsdfClosure for DiffuseBsdfClosure {
    fn flags(&self) -> BsdfFlags {
        BsdfFlags::DIFFUSE_REFLECTION
    }
    fn evaluate(&self, wo: Vec3, wi: Vec3) -> SampledSpectrum {
        let r = self.color;
        if Frame::same_hemisphere(wo, wi) {
            r * FRAC_1_PI
        } else {
            SampledSpectrum::zero()
        }
    }
    fn evaluate_pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if Frame::same_hemisphere(wo, wi) {
            Frame::abs_cos_theta(wi) * FRAC_1_PI
        } else {
            0.0
        }
    }
    fn sample(&self, u: Vec2, wo: Vec3) -> Option<BsdfSample> {
        let r = self.color;

        let wi = {
            let w = consine_hemisphere_sampling(u);
            if Frame::same_hemisphere(w, wo) {
                w
            } else {
                vec3(w.x, -w.y, w.z)
            }
        };
        Some(BsdfSample {
            f: r * FRAC_1_PI,
            wi,
            pdf: Frame::abs_cos_theta(wi) * FRAC_1_PI,
            flag: self.flags(),
        })
    }
}

pub struct SpecularBsdf {
    pub color: Arc<dyn SpectrumTexture>,
}

pub struct SpecularBsdfClosure {
    pub color: SampledSpectrum,
}
impl LocalBsdfClosure for SpecularBsdfClosure {
    fn flags(&self) -> BsdfFlags {
        BsdfFlags::SPECULAR_REFLECTION
    }
    fn evaluate(&self, _wo: Vec3, _wi: Vec3) -> SampledSpectrum {
        SampledSpectrum::zero()
    }
    fn evaluate_pdf(&self, _wo: Vec3, _wi: Vec3) -> f32 {
        0.0
    }
    fn sample(&self, _u: Vec2, wo: Vec3) -> Option<BsdfSample> {
        let r = self.color;

        let wi = reflect(wo, vec3(0.0, 1.0, 0.0));
        Some(BsdfSample {
            f: r / Frame::abs_cos_theta(wi),
            wi,
            pdf: 1.0,
            flag: self.flags(),
        })
    }
}
pub struct FresnelSpecularBsdf {
    pub kr: Arc<dyn SpectrumTexture>,
    pub kt: Arc<dyn SpectrumTexture>,
    pub a: f32,
    pub b: f32,
    // pub ior: Arc<dyn SpectrumTexture>,
}
pub struct FresnelSpecularBsdfClosure {
    kt: SampledSpectrum,
    kr: SampledSpectrum,
    eta_a: f32,
    eta_b: f32,
    mode: TransportMode,
}
impl Bsdf for FresnelSpecularBsdf {
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        mode: TransportMode,
        lambda: &mut SampledWavelengths,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        if self.b > 0.0 {
            lambda.terminate_secondary();
        }
        let primary = lambda[0];
        let ior = self.a + self.b / (primary * 1e-3).powi(2);
        arena.alloc(FresnelSpecularBsdfClosure {
            kt: self.kt.evaluate(sp, lambda),
            kr: self.kr.evaluate(sp, lambda),
            eta_a: 1.0,
            eta_b: ior,
            mode,
        })
    }
}
impl LocalBsdfClosure for FresnelSpecularBsdfClosure {
    fn flags(&self) -> BsdfFlags {
        BsdfFlags::SPECULAR_REFLECTION | BsdfFlags::SPECULAR_REFRACTION
    }
    fn evaluate(&self, _wo: Vec3, _wi: Vec3) -> SampledSpectrum {
        SampledSpectrum::zero()
    }

    fn evaluate_pdf(&self, _wo: Vec3, _wi: Vec3) -> f32 {
        0.0
    }

    fn sample(&self, u: Vec2, wo: Vec3) -> Option<BsdfSample> {
        let f = fr_dielectric(Frame::cos_theta(wo), self.eta_a, self.eta_b);
        if u[0] < f {
            let wi = vec3(-wo.x, wo.y, -wo.z);
            Some(BsdfSample {
                flag: BsdfFlags::SPECULAR_REFLECTION,
                wi,
                pdf: f,
                f: self.kr * f / Frame::abs_cos_theta(wi),
            })
        } else {
            let entering = Frame::cos_theta(wo) > 0.0;
            let (eta_i, eta_t) = if entering {
                (self.eta_a, self.eta_b)
            } else {
                (self.eta_b, self.eta_a)
            };
            let wi = refract(wo, Vec3::Y, self.eta_b / self.eta_a)?;
            let mut ft = self.kt * (1.0 - f);
            if self.mode == TransportMode::CameraToLight {
                ft *= (eta_i * eta_i) / (eta_t * eta_t);
            }
            Some(BsdfSample {
                wi,
                pdf: 1.0 - f,
                f: ft / Frame::abs_cos_theta(wi),
                flag: BsdfFlags::SPECULAR_REFRACTION,
            })
        }
    }
}

pub struct GPUBsdfProxy {
    pub color: Arc<dyn SpectrumTexture>,
    pub metallic: Arc<dyn FloatTexture>,
    pub roughness: Arc<dyn FloatTexture>,
    pub emission: Arc<dyn SpectrumTexture>,
}

impl Bsdf for GPUBsdfProxy {
    fn emission(&self) -> Option<Arc<dyn SpectrumTexture>> {
        Some(self.emission.clone())
    }
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        _sp: &ShadingPoint,
        _mode: TransportMode,
        _lambda: &mut SampledWavelengths,
        _arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        panic!("shouldn't be called on cpu")
    }
}

// mod test {
//     use crate::bsdf::BsdfTester;

//     #[test]
//     fn test_bsdf() {
//         use super::DiffuseBsdf;
//         use crate::ltc::GgxLtcBsdf;
//         use crate::texture::ConstantTexture;
//         use std::sync::Arc;
//         let ltc = GgxLtcBsdf {
//             roughness: Arc::new(ConstantTexture::<f32> { value: 0.1 }),
//             color: Arc::new(ConstantTexture::<f32> { value: 0.5 }),
//         };
//         // let ltc = DiffuseBsdf {
//         //     color: Arc::new(ConstantTexture::<f32> { value: 0.1 }),
//         // };
//         let tester = BsdfTester { bsdf: ltc };
//         let pi = tester.run();
//         assert!(
//             (pi - 2.0 * crate::PI).abs() < 1e-2,
//             "integral={}, 2pi={}",
//             pi,
//             2.0 * crate::PI
//         );
//     }
// }
