use std::sync::Arc;

use crate::{
    bsdf::{Bsdf, BsdfFlags, BsdfSample, LocalBsdfClosure, SpecularBsdfClosure},
    texture::{FloatTexture, ShadingPoint, SpectrumTexture},
    *,
};
use akari_common::glam::vec3a;
use akari_const::GGX_LTC_FIT;
use bumpalo::Bump;
use glam::DMat3;

use super::TransportMode;

pub struct GgxLtcBsdf {
    pub color: Arc<dyn SpectrumTexture>,
    pub roughness: Arc<dyn FloatTexture>,
}
pub struct GgxLtcBsdfClosure {
    pub color: SampledSpectrum,
    pub roughness: f32,
}
pub struct LTC {
    mat: DMat3,
    amp: f64,
}

impl LTC {
    pub fn from_theta_alpha(cos_theta: f32, alpha: f32) -> LTC {
        const SIZE: usize = 64;
        let t = (((1.0 - cos_theta).sqrt() * SIZE as f32).floor().max(0.0) as usize).min(SIZE - 1);
        let a = ((alpha.sqrt() * SIZE as f32).floor().max(0.0) as usize).min(SIZE - 1);
        let m = &GGX_LTC_FIT.mat[a + t * SIZE];
        Self {
            mat: glam::dmat3(
                glam::dvec3(m[0] as f64, m[1] as f64, m[2] as f64),
                glam::dvec3(m[3] as f64, m[4] as f64, m[5] as f64),
                glam::dvec3(m[6] as f64, m[7] as f64, m[8] as f64),
            )
            .transpose(),
            amp: GGX_LTC_FIT.amp[a + t * SIZE] as f64,
        }
    }
    #[allow(non_snake_case)]
    pub fn eval_f_pdf(&self, w: Vec3A) -> (f32, f32) {
        use glam::Vec3Swizzles;
        let w = w.xzy().as_dvec3();
        let inv_m = self.mat.inverse();

        let w_original = (inv_m * w).normalize();
        let w_ = self.mat * w_original;

        let l = w_.length();
        let jacobian = self.mat.determinant().abs() / (l * l * l);

        let D = 1.0 / PI as f64 * w_original.z.max(0.0);

        let f = self.amp * D / jacobian;

        ((f / w.z.abs()) as f32, (D / jacobian) as f32)
    }
    // pub fn pdf(&self, w)
    pub fn sample(&self, u1: f32, u2: f32) -> Vec3A {
        use glam::Vec3Swizzles;
        let w = consine_hemisphere_sampling(vec2(u1, u2)).xzy().as_dvec3();
        (self.mat * w).normalize().xzy().as_vec3().into()
    }
}

#[allow(non_snake_case)]
fn frame_from_wo(wo: Vec3A) -> Frame {
    let N = vec3a(0.0, 1.0, 0.0);
    let T = (wo - N * wo.dot(N)).normalize();
    let B = N.cross(T).normalize();
    // println!("{:?} {:?} {:?}",T, N, B);
    Frame { T, B, N }
}

impl LocalBsdfClosure for GgxLtcBsdfClosure {
    fn flags(&self)->BsdfFlags{
        BsdfFlags::GLOSSY_REFLECTION
    }
    fn evaluate(&self, wo: Vec3A, wi: Vec3A) -> SampledSpectrum {
        if !Frame::same_hemisphere(wo, wi) {
            return SampledSpectrum::zero();
        }
        let theta = Frame::abs_cos_theta(wo);
        let alpha = self.roughness.powi(2);
        let color = self.color;
        let ltc = LTC::from_theta_alpha(theta, alpha);
        let frame = frame_from_wo(wo);
        let f = ltc
            .eval_f_pdf(frame.to_local(vec3a(wi.x, wi.y.abs(), wi.z)))
            .0;
        if f.is_nan() {
            SampledSpectrum::zero()
        } else {
            color * f
        }
    }

    fn evaluate_pdf(&self, wo: Vec3A, wi: Vec3A) -> f32 {
        if !Frame::same_hemisphere(wo, wi) {
            return 0.0;
        }
        let theta = Frame::abs_cos_theta(wo);
        let alpha = self.roughness.powi(2);
        let ltc = LTC::from_theta_alpha(theta, alpha);
        let frame = frame_from_wo(wo);
        let pdf = ltc
            .eval_f_pdf(frame.to_local(vec3a(wi.x, wi.y.abs(), wi.z)))
            .1;
        if !pdf.is_nan() {
            pdf
        } else {
            0.0
        }
    }

    fn sample(&self, u: Vec2, wo: Vec3A) -> Option<BsdfSample> {
        let theta = Frame::abs_cos_theta(wo);
        let alpha = self.roughness.powi(2);
        let ltc = LTC::from_theta_alpha(theta, alpha);
        let mut wi = ltc.sample(u.x, u.y);
        let frame = frame_from_wo(wo);
        let color = self.color;
        let (f, pdf) = ltc.eval_f_pdf(wi);
        if !(pdf > 0.0) && !(f > 0.0) {
            return None;
        }
        wi = frame.to_world(wi);
        // println!("{} {} {:?} {:?}", theta, alpha, ltc.mat, wi_);
        // println!("{} {} {:?} {:?} {:?}",theta, alpha, wo,  frame.T, frame.B);
        // println!("{} {} {:?} {:?} {:?}",theta, alpha, wo,  wi, wi_);
        if !Frame::same_hemisphere(wo, wi) {
            wi.y = -wi.y;
        }
        Some(BsdfSample {
            wi,
            f: color * f,
            pdf,
            flag: self.flags(),
        })
    }
}
impl Bsdf for GgxLtcBsdf {
    fn evaluate<'a, 'b: 'a>(
        &'b self,
        sp: &ShadingPoint,
        _mode:TransportMode,
        lambda: &mut SampledWavelengths,
        arena: &'a Bump,
    ) -> &'a dyn LocalBsdfClosure {
        let roughness = self.roughness.evaluate(sp);
        let color = self.color.evaluate(sp, lambda);
        if roughness >= 0.1 {
            arena.alloc(GgxLtcBsdfClosure { color, roughness })
        } else {
            arena.alloc(SpecularBsdfClosure { color })
        }
    }
}
