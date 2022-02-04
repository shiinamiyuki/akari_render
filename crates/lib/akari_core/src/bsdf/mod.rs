use std::sync::Arc;

use bumpalo::Bump;

use crate::sampler::SobolSampler;
use crate::texture::{ShadingPoint, Texture};
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
pub struct BsdfSample {
    pub wi: Vec3,
    pub f: Spectrum,
    pub pdf: f32,
    pub flag: BsdfFlags,
}
#[derive(Clone, Copy)]
pub struct BsdfInfo {
    pub albedo: Spectrum,
    pub roughness: f32,
    pub metallic: f32,
}

pub trait Bsdf: Sync + Send + Base {
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure;
    fn emission(&self) -> Option<Arc<dyn Texture>> {
        None
    }
}
pub trait LocalBsdfClosure: Sync + Send {
    fn evaluate(&self, wo: Vec3, wi: Vec3) -> Spectrum;
    fn evaluate_pdf(&self, wo: Vec3, wi: Vec3) -> f32;
    fn sample(&self, u: Vec2, wo: Vec3) -> Option<BsdfSample>;
    fn info(&self) -> BsdfInfo;
    fn emission(&self) -> Option<&dyn Texture> {
        None
    }
}
#[derive(Copy, Clone)]
pub struct BsdfClosure<'a> {
    pub frame: Frame,
    pub closure: &'a dyn LocalBsdfClosure,
}

impl<'a> BsdfClosure<'a> {
    pub fn evaluate(&self, wo: Vec3, wi: Vec3) -> Spectrum {
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
    pub fn info(&self) -> BsdfInfo {
        self.closure.info()
    }
    pub fn emission(&self) -> Option<Arc<dyn Texture>> {
        None
    }
}
pub struct EmissiveBsdf {
    pub base: Arc<dyn Bsdf>,
    pub emission: Arc<dyn Texture>,
}
impl_base!(EmissiveBsdf);
pub struct EmissiveBsdfClosure<'a> {
    pub base: &'a dyn LocalBsdfClosure,
    pub emission: &'a dyn Texture,
}
impl<'a> LocalBsdfClosure for EmissiveBsdfClosure<'a> {
    fn evaluate(&self, wo: Vec3, wi: Vec3) -> Spectrum {
        self.base.evaluate(wo, wi)
    }
    fn evaluate_pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        self.base.evaluate_pdf(wo, wi)
    }
    fn sample(&self, u: Vec2, wo: Vec3) -> Option<BsdfSample> {
        self.base.sample(u, wo)
    }
    fn info(&self) -> BsdfInfo {
        self.base.info()
    }
    fn emission(&self) -> Option<&dyn Texture> {
        Some(self.emission)
    }
}

impl Bsdf for EmissiveBsdf {
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        arena.alloc(EmissiveBsdfClosure {
            base: self.base.evaluate(sp, arena),
            emission: self.emission.as_ref(),
        })
    }
    fn emission(&self) -> Option<Arc<dyn Texture>> {
        Some(self.emission.clone())
    }
}
pub struct MixBsdf<A: Bsdf, B: Bsdf> {
    pub bsdf_a: A,
    pub bsdf_b: B,
    pub frac: Arc<dyn Texture>,
}

impl<A, B> Base for MixBsdf<A, B>
where
    A: Bsdf + 'static,
    B: Bsdf + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}
impl<A, B> Bsdf for MixBsdf<A, B>
where
    A: Bsdf + 'static,
    B: Bsdf + 'static,
{
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        arena.alloc(MixBsdfClosure {
            bsdf_a: self.bsdf_a.evaluate(sp, arena),
            bsdf_b: self.bsdf_b.evaluate(sp, arena),
            frac: self.frac.evaluate_f(sp),
        })
    }
}
pub struct MixBsdfClosure<'a> {
    pub bsdf_a: &'a dyn LocalBsdfClosure,
    pub bsdf_b: &'a dyn LocalBsdfClosure,
    pub frac: f32,
}
impl<'a> LocalBsdfClosure for MixBsdfClosure<'a> {
    fn evaluate(&self, wo: Vec3, wi: Vec3) -> Spectrum {
        Spectrum::lerp(
            self.bsdf_a.evaluate(wo, wi),
            self.bsdf_b.evaluate(wo, wi),
            self.frac,
        )
    }
    fn evaluate_pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        lerp_scalar(
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
                if sample.flag.intersects(BsdfFlags::SPECULAR) {
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
                if sample.flag.intersects(BsdfFlags::SPECULAR) {
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
    fn info(&self) -> BsdfInfo {
        let info_a = self.bsdf_a.info();
        let info_b = self.bsdf_b.info();
        let frac = self.frac;
        BsdfInfo {
            roughness: lerp_scalar(info_a.roughness, info_b.roughness, frac),
            albedo: Spectrum::lerp(info_a.albedo, info_b.albedo, frac),
            metallic: lerp_scalar(info_a.metallic, info_b.metallic, frac),
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
//     color: Spectrum,
//     roughness: f32,
//     sheen: f32,
//     tint: Spectrum,
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
//     fn evaluate_tint(color: Spectrum) -> Spectrum {
//         let rgb = color.to_rgb_linear();
//         let luminance = vec3(0.3, 0.6, 0.1).dot(rgb);
//         if luminance > 0.0 {
//             color * (1.0 / luminance)
//         } else {
//             Spectrum::from_rgb_linear(vec3(1.0, 1.0, 1.0))
//         }
//     }
//     fn evaluate_sheen(&self, sp: &ShadingPoint, wo: Vec3, wi: Vec3, wm: Vec3) -> Spectrum {
//         let sheen = self.sheen;
//         if sheen <= 0.0 {
//             return Spectrum::zero();
//         }
//         let dot_hl = wm.dot(wi);
//         let tint = Self::evaluate_tint(self.color);
//         Spectrum::lerp(Spectrum::one(), tint, self.sheen_tint) * self.sheen * schlick_weight(dot_hl)
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
    pub color: Arc<dyn Texture>,
}
impl_base!(DiffuseBsdf);
impl Bsdf for DiffuseBsdf {
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        arena.alloc(DiffuseBsdfClosure {
            color: self.color.evaluate_s(sp),
        })
    }
}
pub struct DiffuseBsdfClosure {
    pub color: Spectrum,
}
impl LocalBsdfClosure for DiffuseBsdfClosure {
    fn info(&self) -> BsdfInfo {
        BsdfInfo {
            roughness: 1.0,
            albedo: self.color,
            metallic: 0.0,
        }
    }
    fn evaluate(&self, wo: Vec3, wi: Vec3) -> Spectrum {
        let r = self.color;
        if Frame::same_hemisphere(wo, wi) {
            r * FRAC_1_PI
        } else {
            Spectrum::zero()
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
            flag: BsdfFlags::DIFFUSE_REFLECTION,
        })
    }
}

pub struct SpecularBsdf {
    pub color: Arc<dyn Texture>,
}
impl_base!(SpecularBsdf);
pub struct SpecularBsdfClosure {
    pub color: Spectrum,
}
impl LocalBsdfClosure for SpecularBsdfClosure {
    fn info(&self) -> BsdfInfo {
        BsdfInfo {
            roughness: 0.0,
            albedo: self.color,
            metallic: 1.0,
        }
    }
    fn evaluate(&self, _wo: Vec3, _wi: Vec3) -> Spectrum {
        Spectrum::zero()
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
            flag: BsdfFlags::SPECULAR_REFLECTION,
        })
    }
}

pub struct GPUBsdfProxy {
    pub color: Arc<dyn Texture>,
    pub metallic: Arc<dyn Texture>,
    pub roughness: Arc<dyn Texture>,
    pub emission: Arc<dyn Texture>,
}
impl_base!(GPUBsdfProxy);
impl Bsdf for GPUBsdfProxy {
    fn emission(&self) -> Option<Arc<dyn Texture>> {
        Some(self.emission.clone())
    }
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        arena: &'a Bump,
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
