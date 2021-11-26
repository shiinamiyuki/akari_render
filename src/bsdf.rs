use crate::texture::{ShadingPoint, Texture};
use crate::*;

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
    pub pdf: Float,
    pub flag: BsdfFlags,
}
#[derive(Clone, Copy)]
pub struct BsdfInfo {
    pub albedo: Spectrum,
    pub roughness: Float,
    pub metallic: Float,
}

pub trait Bsdf: Sync + Send + AsAny {
    fn evaluate(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Spectrum;
    fn evaluate_pdf(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Float;
    fn sample(&self, sp: &ShadingPoint, u: &Vec2, wo: &Vec3) -> Option<BsdfSample>;
    fn info(&self, sp: &ShadingPoint) -> BsdfInfo;
    fn emission(&self) -> Option<Arc<dyn Texture>> {
        None
    }
}

#[derive(Copy, Clone)]
pub struct BsdfClosure<'a> {
    pub frame: Frame,
    pub bsdf: &'a dyn Bsdf,
    pub sp: ShadingPoint,
}
impl<'a> BsdfClosure<'a> {
    pub fn evaluate(&self, wo: &Vec3, wi: &Vec3) -> Spectrum {
        self.bsdf.evaluate(
            &self.sp,
            &self.frame.to_local(&wo),
            &self.frame.to_local(&wi),
        )
    }
    pub fn evaluate_pdf(&self, wo: &Vec3, wi: &Vec3) -> Float {
        self.bsdf.evaluate_pdf(
            &self.sp,
            &self.frame.to_local(&wo),
            &self.frame.to_local(&wi),
        )
    }
    pub fn sample(&self, u: &Vec2, wo: &Vec3) -> Option<BsdfSample> {
        let mut sample = self.bsdf.sample(&self.sp, &u, &self.frame.to_local(&wo))?;
        sample.wi = self.frame.to_world(&sample.wi);
        Some(sample)
    }
}

pub struct NullBsdf {}
impl Bsdf for NullBsdf {
    fn evaluate(&self, _sp: &ShadingPoint, _wo: &Vec3, _wi: &Vec3) -> Spectrum {
        Spectrum::zero()
    }
    fn evaluate_pdf(&self, _sp: &ShadingPoint, _wo: &Vec3, _wi: &Vec3) -> Float {
        0.0
    }
    fn sample(&self, _sp: &ShadingPoint, _u: &Vec2, _wo: &Vec3) -> Option<BsdfSample> {
        None
    }
    fn info(&self, _sp: &ShadingPoint) -> BsdfInfo {
        BsdfInfo {
            roughness: 1.0,
            albedo: Spectrum::zero(),
            metallic: 0.0,
        }
    }
}
impl_as_any!(NullBsdf);
pub struct EmissiveBsdf {
    pub base: Arc<dyn Bsdf>,
    pub emission: Arc<dyn Texture>,
}
impl Bsdf for EmissiveBsdf {
    fn evaluate(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Spectrum {
        self.base.evaluate(sp, wo, wi)
    }
    fn evaluate_pdf(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Float {
        self.base.evaluate_pdf(sp, wo, wi)
    }
    fn sample(&self, sp: &ShadingPoint, u: &Vec2, wo: &Vec3) -> Option<BsdfSample> {
        self.base.sample(sp, u, wo)
    }
    fn info(&self, sp: &ShadingPoint) -> BsdfInfo {
        self.base.info(sp)
    }
    fn emission(&self) -> Option<Arc<dyn Texture>> {
        Some(self.emission.clone())
    }
}
impl_as_any!(EmissiveBsdf);
pub struct MixBsdf<A: Bsdf, B: Bsdf> {
    pub bsdf_a: A,
    pub bsdf_b: B,
    pub frac: Arc<dyn Texture>,
}
impl<A, B> AsAny for MixBsdf<A, B>
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
}
impl<A, B> Bsdf for MixBsdf<A, B>
where
    A: Bsdf + 'static,
    B: Bsdf + 'static,
{
    fn evaluate(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Spectrum {
        let frac = self.frac.evaluate_f(sp);
        Spectrum::lerp(
            &self.bsdf_a.evaluate(sp, wo, wi),
            &self.bsdf_b.evaluate(sp, wo, wi),
            frac,
        )
    }
    fn evaluate_pdf(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Float {
        let frac = self.frac.evaluate_f(sp);
        glm::lerp_scalar(
            self.bsdf_a.evaluate_pdf(sp, wo, wi),
            self.bsdf_b.evaluate_pdf(sp, wo, wi),
            frac,
        )
    }
    fn sample(&self, sp: &ShadingPoint, u: &Vec2, wo: &Vec3) -> Option<BsdfSample> {
        let frac = self.frac.evaluate_f(sp);
        let prob = (1.0 - frac).clamp(0.0000001, 0.9999999);
        if u[0] < prob {
            let remapped_u = vec2(u[0] / prob, u[1]);
            if let Some(sample) = self.bsdf_a.sample(sp, &remapped_u, wo) {
                if sample.flag.intersects(BsdfFlags::SPECULAR) {
                    Some(BsdfSample {
                        pdf: sample.pdf * prob,
                        ..sample
                    })
                } else {
                    Some(BsdfSample {
                        flag: sample.flag,
                        wi: sample.wi,
                        pdf: self.evaluate_pdf(sp, wo, &sample.wi),
                        f: self.evaluate(sp, wo, &sample.wi),
                    })
                }
            } else {
                None
            }
        } else {
            let remapped_u = vec2((u[0] - prob) / (1.0 - prob), u[1]);
            if let Some(sample) = self.bsdf_b.sample(sp, &remapped_u, wo) {
                if sample.flag.intersects(BsdfFlags::SPECULAR) {
                    Some(BsdfSample {
                        pdf: sample.pdf * (1.0 - prob),
                        ..sample
                    })
                } else {
                    Some(BsdfSample {
                        flag: sample.flag,
                        wi: sample.wi,
                        pdf: self.evaluate_pdf(sp, wo, &sample.wi),
                        f: self.evaluate(sp, wo, &sample.wi),
                    })
                }
            } else {
                None
            }
        }
    }
    fn info(&self, sp: &ShadingPoint) -> BsdfInfo {
        let info_a = self.bsdf_a.info(sp);
        let info_b = self.bsdf_b.info(sp);
        let frac = self.frac.evaluate_f(sp);
        BsdfInfo {
            roughness: glm::lerp_scalar(info_a.roughness, info_b.roughness, frac),
            albedo: Spectrum::lerp(&info_a.albedo, &info_b.albedo, frac),
            metallic: glm::lerp_scalar(info_a.metallic, info_b.metallic, frac),
        }
    }
}
pub struct DisneyBsdf {
    pub color: Arc<dyn Texture>,
    pub roughness: Arc<dyn Texture>,
    pub tint: Arc<dyn Texture>,
    pub sheen: Arc<dyn Texture>,
}
struct DisneyBsdfClosure {
    color: Spectrum,
    roughness: Float,
    sheen: Float,
    tint: Spectrum,
    sheen_tint: Float,
}
pub fn schlick_weight(cos_theta: Float) -> Float {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    let m2 = m * m;
    m2 * m2 * m
}
pub fn f_schlick_(f0: Float, cos_theta: Float) -> Float {
    f0 + (1.0 - f0) * (1.0 - cos_theta).powi(5)
}
impl DisneyBsdfClosure {
    fn gtr1(dot_hl: Float, a: Float) -> Float {
        if a >= 1.0 {
            FRAC_1_PI
        } else {
            let a2 = a * a;
            (a2 - 1.0) / (PI * a2.ln() * (1.0 + (a2 - 1.0) * dot_hl * dot_hl))
        }
    }
    fn separable_ggx_g1(w: &Vec3, a: Float) -> Float {
        let a2 = a * a;
        let cos_theta = Frame::abs_cos_theta(w);
        1.0 / (cos_theta + (a2 + cos_theta - a2 * cos_theta * cos_theta).sqrt())
    }
    fn evaluate_clearcoat(
        clearcoat: Float,
        alpha: Float,
        wo: &Vec3,
        wi: &Vec3,
        wh: &Vec3,
    ) -> Float {
        if clearcoat <= 0.0 {
            return 0.0;
        }
        let abs_dot_nh = Frame::abs_cos_theta(wh);
        let abs_dot_hl = Frame::abs_cos_theta(wi);
        let abs_dot_hv = Frame::abs_cos_theta(wo);
        let dot_hl = glm::dot(wh, wi);
        let d = Self::gtr1(abs_dot_nh, glm::lerp_scalar(0.1, 0.001, alpha));
        let f = f_schlick_(0.04, dot_hl);
        let gl = Self::separable_ggx_g1(wi, 0.25);
        let gv = Self::separable_ggx_g1(wo, 0.25);

        0.25 * clearcoat * d * f * gl * gv
    }
    fn evaluate_tint(color: &Spectrum) -> Spectrum {
        let rgb = color.to_rgb_linear();
        let luminance = glm::dot(&vec3(0.3, 0.6, 0.1), &rgb);
        if luminance > 0.0 {
            *color * (1.0 / luminance)
        } else {
            Spectrum::from_rgb_linear(&vec3(1.0, 1.0, 1.0))
        }
    }
    fn evaluate_sheen(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3, wm: &Vec3) -> Spectrum {
        let sheen = self.sheen;
        if sheen <= 0.0 {
            return Spectrum::zero();
        }
        let dot_hl = glm::dot(wm, wi);
        let tint = Self::evaluate_tint(&self.color);
        Spectrum::lerp(&Spectrum::one(), &tint, self.sheen_tint)
            * self.sheen
            * schlick_weight(dot_hl)
    }
    fn ggx_aniso_d(wh: &Vec3, ax: Float, ay: Float) -> Float {
        let dot_hx2 = wh.x * wh.x;
        let dot_hy2 = wh.y * wh.y;
        let cos2 = Frame::cos2_theta(wh);
        let ax2 = ax * ax;
        let ay2 = ay * ay;
        1.0 / (PI * ax * ay * (dot_hx2 / ax2 + dot_hy2 / ay2 + cos2).powi(2))
    }
    fn separable_smith_ggx_g1(w: &Vec3, wh: &Vec3, ax: Float, ay: Float) -> Float {
        let dot_hw = glm::dot(wh, w);
        if dot_hw <= 0.0 {
            return 0.0;
        }
        let abs_tan_theta = Frame::tan_theta(w).abs();
        if abs_tan_theta.is_infinite() {
            return 0.0;
        }
        let a = (Frame::cos2_phi(w) * ax * ax + Frame::sin2_phi(w) * ay * ay).sqrt();
        let a2_tan = (a * abs_tan_theta).powi(2);
        let lambda = 0.5 * (-1.0 + (1.0 + a2_tan).sqrt());
        1.0 / (1.0 + lambda)
    }
}

pub struct DiffuseBsdf {
    pub color: Arc<dyn Texture>,
}
impl_as_any!(DiffuseBsdf);
impl Bsdf for DiffuseBsdf {
    fn info(&self, sp: &ShadingPoint) -> BsdfInfo {
        BsdfInfo {
            roughness: 1.0,
            albedo: self.color.evaluate_s(sp),
            metallic: 0.0,
        }
    }
    fn evaluate(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Spectrum {
        let r = self.color.evaluate_s(sp);
        if Frame::same_hemisphere(&wo, &wi) {
            r * FRAC_1_PI
        } else {
            Spectrum::zero()
        }
    }
    fn evaluate_pdf(&self, _sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Float {
        if Frame::same_hemisphere(&wo, &wi) {
            Frame::abs_cos_theta(&wi) * FRAC_1_PI
        } else {
            0.0
        }
    }
    fn sample(&self, sp: &ShadingPoint, u: &Vec2, wo: &Vec3) -> Option<BsdfSample> {
        let r = self.color.evaluate_s(sp);

        let wi = {
            let w = consine_hemisphere_sampling(&u);
            if Frame::same_hemisphere(&w, &wo) {
                w
            } else {
                vec3(w.x, -w.y, w.z)
            }
        };
        Some(BsdfSample {
            f: r * FRAC_1_PI,
            wi,
            pdf: Frame::abs_cos_theta(&wi) * FRAC_1_PI,
            flag: BsdfFlags::DIFFUSE_REFLECTION,
        })
    }
}

pub struct SpecularBsdf {
    pub color: Arc<dyn Texture>,
}
impl_as_any!(SpecularBsdf);
impl Bsdf for SpecularBsdf {
    fn info(&self, sp: &ShadingPoint) -> BsdfInfo {
        BsdfInfo {
            roughness: 0.0,
            albedo: self.color.evaluate_s(sp),
            metallic: 1.0,
        }
    }
    fn evaluate(&self, sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Spectrum {
        Spectrum::zero()
    }
    fn evaluate_pdf(&self, _sp: &ShadingPoint, wo: &Vec3, wi: &Vec3) -> Float {
        0.0
    }
    fn sample(&self, sp: &ShadingPoint, u: &Vec2, wo: &Vec3) -> Option<BsdfSample> {
        let r = self.color.evaluate_s(sp);

        let wi = reflect(wo, &vec3(0.0, 1.0, 0.0));
        Some(BsdfSample {
            f: r / Frame::abs_cos_theta(&wi),
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
impl_as_any!(GPUBsdfProxy);
impl Bsdf for GPUBsdfProxy {
    fn emission(&self) -> Option<Arc<dyn Texture>> {
        Some(self.emission.clone())
    }
    fn evaluate(&self, _sp: &ShadingPoint, _wo: &Vec3, _wi: &Vec3) -> Spectrum {
        panic!("shouldn't be called on cpu")
    }
    fn evaluate_pdf(&self, _sp: &ShadingPoint, _wo: &Vec3, _wi: &Vec3) -> Float {
        panic!("shouldn't be called on cpu")
    }
    fn sample(&self, _sp: &ShadingPoint, _u: &Vec2, _wo: &Vec3) -> Option<BsdfSample> {
        panic!("shouldn't be called on cpu")
    }
    fn info(&self, _sp: &ShadingPoint) -> BsdfInfo {
        panic!("shouldn't be called on cpu")
    }
}
