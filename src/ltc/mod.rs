use crate::{
    bsdf::{Bsdf, BsdfFlags, BsdfSample},
    ltc::fit::GGX_LTC_FIT,
    texture::Texture,
    *,
};
use lazy_static::lazy_static;
mod fit;
#[allow(dead_code)]
pub struct GgxLtcfit {
    pub mat: [[Float; 9]; 64 * 64],
    pub amp: [Float; 64 * 64],
}

pub struct GgxLtcBsdf {
    pub color: Arc<dyn Texture>,
    pub roughness: Arc<dyn Texture>,
}

pub struct LTC {
    mat: glm::DMat3,
    amp: f64,
}

impl LTC {
    pub fn from_theta_alpha(cos_theta: Float, alpha: Float) -> LTC {
        const SIZE: usize = 64;
        let t =
            (((1.0 - cos_theta).sqrt() * SIZE as Float).floor().max(0.0) as usize).min(SIZE - 1);
        let a = ((alpha.sqrt() * SIZE as Float).floor().max(0.0) as usize).min(SIZE - 1);
        let m = &GGX_LTC_FIT.mat[a + t * SIZE];
        Self {
            mat: glm::DMat3::new(
                m[0] as f64,
                m[1] as f64,
                m[2] as f64,
                m[3] as f64,
                m[4] as f64,
                m[5] as f64,
                m[6] as f64,
                m[7] as f64,
                m[8] as f64,
            ),
            amp: GGX_LTC_FIT.amp[a + t * SIZE] as f64,
        }
    }
    #[allow(non_snake_case)]
    pub fn eval_f_pdf(&self, w: &Vec3) -> (Float, Float) {
        let w = w.xzy().cast::<f64>();
        let inv_m = glm::inverse(&self.mat);

        let w_original = glm::normalize(&(inv_m * w));
        let w_ = self.mat * w_original;

        let l = glm::length(&w_);
        let jacobian = glm::determinant(&self.mat).abs() / (l * l * l);

        let D = 1.0 / PI as f64 * w_original.z.max(0.0);

        let f = self.amp * D / jacobian;

        ((f / w.z.abs()) as Float, (D / jacobian) as Float)
    }
    // pub fn pdf(&self, w)
    pub fn sample(&self, u1: Float, u2: Float) -> Vec3 {
        let w = consine_hemisphere_sampling(&vec2(u1, u2))
            .xzy()
            .cast::<f64>();
        glm::normalize(&(self.mat * w)).xzy().cast::<Float>()
    }
}

#[allow(non_snake_case)]
fn frame_from_wo(wo: &Vec3) -> Frame {
    let N = vec3(0.0, 1.0, 0.0);
    let T = glm::normalize(&(wo - N * glm::dot(wo, &N)));
    let B = glm::normalize(&glm::cross(&N, &T));
    // println!("{:?} {:?} {:?}",T, N, B);
    Frame { T, B, N }
}
impl_as_any!(GgxLtcBsdf);
impl Bsdf for GgxLtcBsdf {
    fn evaluate(&self, sp: &texture::ShadingPoint, wo: &Vec3, wi: &Vec3) -> Spectrum {
        if !Frame::same_hemisphere(wo, wi) {
            return Spectrum::zero();
        }
        let theta = Frame::abs_cos_theta(wo);
        let alpha = self.roughness.evaluate_f(sp).powi(2);
        let color = self.color.evaluate_s(sp);
        let ltc = LTC::from_theta_alpha(theta, alpha);
        let frame = frame_from_wo(wo);
        let f = ltc
            .eval_f_pdf(&frame.to_local(&vec3(wi.x, wi.y.abs(), wi.z)))
            .0;
        if f.is_nan() {
            Spectrum::zero()
        } else {
            color * f
        }
    }

    fn evaluate_pdf(&self, sp: &texture::ShadingPoint, wo: &Vec3, wi: &Vec3) -> Float {
        if !Frame::same_hemisphere(wo, wi) {
            return 0.0;
        }
        let theta = Frame::abs_cos_theta(wo);
        let alpha = self.roughness.evaluate_f(sp).powi(2);
        let ltc = LTC::from_theta_alpha(theta, alpha);
        let frame = frame_from_wo(wo);
        let pdf = ltc
            .eval_f_pdf(&frame.to_local(&vec3(wi.x, wi.y.abs(), wi.z)))
            .1;
        if !pdf.is_nan() {
            pdf
        } else {
            0.0
        }
    }

    fn sample(&self, sp: &texture::ShadingPoint, u: &Vec2, wo: &Vec3) -> Option<BsdfSample> {
        let theta = Frame::abs_cos_theta(wo);
        let alpha = self.roughness.evaluate_f(sp).powi(2);
        let ltc = LTC::from_theta_alpha(theta, alpha);
        let mut wi = ltc.sample(u.x, u.y);
        let frame = frame_from_wo(wo);
        let color = self.color.evaluate_s(sp);
        let (f, pdf) = ltc.eval_f_pdf(&wi);
        if !(pdf > 0.0) && !(f > 0.0) {
            return None;
        }
        wi = frame.to_world(&wi);
        // println!("{} {} {:?} {:?}", theta, alpha, ltc.mat, wi_);
        // println!("{} {} {:?} {:?} {:?}",theta, alpha, wo,  frame.T, frame.B);
        // println!("{} {} {:?} {:?} {:?}",theta, alpha, wo,  wi, wi_);
        if !Frame::same_hemisphere(wo, &wi) {
            wi.y = -wi.y;
        }
        Some(BsdfSample {
            wi,
            f: color * f,
            pdf,
            flag: BsdfFlags::GLOSSY_REFLECTION,
        })
    }

    fn info(&self, sp: &texture::ShadingPoint) -> bsdf::BsdfInfo {
        bsdf::BsdfInfo {
            albedo: self.color.evaluate_s(sp),
            roughness: self.roughness.evaluate_f(sp),
            metallic: 1.0,
        }
    }
}
