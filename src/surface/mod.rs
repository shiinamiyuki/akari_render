use crate::color::{ColorRepr, FlatColor};
use crate::geometry::{face_forward, reflect, refract};
use crate::microfacet::MicrofacetDistribution;
use crate::sampling::weighted_discrete_choice2_and_remap;
use crate::texture::TextureEvaluator;
use crate::{color::Color, geometry::Frame, interaction::SurfaceInteraction, *};

pub struct BsdfEvalContext<'a> {
    pub texture: &'a TextureEvaluator,
    pub color_repr: ColorRepr,
}

pub mod diffuse;
pub mod glass;
pub mod principled;

#[derive(Clone, Aggregate)]
pub struct BsdfSample {
    pub wi: Expr<Float3>,
    pub pdf: Expr<f32>,
    pub color: Color,
    pub valid: Bool,
    pub lobe_roughness: Expr<f32>,
}

pub trait Bsdf {
    // return f(wo, wi) * abs_cos_theta(wi)
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color;
    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample;
    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Float;
    fn albedo(&self, wo: Expr<Float3>, ctx: &BsdfEvalContext) -> Color;
}

pub trait Surface {
    fn closure(&self, si: Expr<SurfaceInteraction>, ctx: &BsdfEvalContext) -> BsdfClosure;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BsdfBlendMode {
    Addictive,
    Mix,
}
pub struct BsdfMixture {
    pub frac: Box<dyn Fn(Expr<Float3>, &BsdfEvalContext) -> Expr<f32>>,
    pub bsdf_a: Box<dyn Bsdf>,
    pub bsdf_b: Box<dyn Bsdf>,
    pub mode: BsdfBlendMode,
}
impl BsdfMixture {
    const EPS: f32 = 1e-4;
}

impl Bsdf for BsdfMixture {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        match self.mode {
            BsdfBlendMode::Addictive => {
                let f_a = self.bsdf_a.evaluate(wo, wi, ctx);
                let f_b = self.bsdf_b.evaluate(wo, wi, ctx);
                f_a + f_b
            }
            BsdfBlendMode::Mix => {
                let frac: Expr<f32> = (self.frac)(wo, ctx);
                let zero = Color::zero(ctx.color_repr);
                let f_a = if_!(frac.cmplt(1.0 - Self::EPS), {
                    self.bsdf_a.evaluate(wo, wi,ctx)
                }, else {
                    zero
                });
                let f_b = if_!(frac.cmpgt(Self::EPS), {
                    self.bsdf_b.evaluate(wo, wi,ctx)
                }, else {
                    zero
                });
                f_a * (1.0 - frac) + f_b * frac
            }
        }
    }

    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        let (which, remapped) =
            weighted_discrete_choice2_and_remap(frac, const_(1u32), const_(0u32), u_select);
        let zero_f = Color::zero(ctx.color_repr);
        let zero_pdf = const_(0.0f32);
        if_!(which.cmpeq(0), {
            let mut sample = self.bsdf_a.sample(wo, remapped,u_sample, ctx);
            let (f_b, pdf_b) = {
                if_!(frac.cmpgt(Self::EPS), {
                    let f_b = self.bsdf_b.evaluate(wo, sample.wi, ctx);
                    let pdf_b = self.bsdf_b.pdf(wo, sample.wi, ctx);
                    (f_b, pdf_b)
                }, else {
                    (zero_f, zero_pdf)
                })
            };
            // cpu_dbg!(make_float2(sample.pdf, pdf_b));
            match self.mode {
                BsdfBlendMode::Addictive => {
                    sample.color = sample.color + f_b;
                },
                BsdfBlendMode::Mix => {
                    sample.color = sample.color * (1.0 - frac) + f_b * frac;
                }
            }

            sample.pdf = sample.pdf * (1.0 - frac) + pdf_b * frac;
            sample
        }, else {
            let mut sample = self.bsdf_b.sample(wo, remapped, u_sample, ctx);
            let (f_a, pdf_a) = {
                if_!(frac.cmplt(1.0 - Self::EPS), {
                    let f_a = self.bsdf_a.evaluate(wo, sample.wi, ctx);
                    let pdf_a = self.bsdf_a.pdf(wo, sample.wi, ctx);
                    (f_a, pdf_a)
                }, else {
                    (zero_f, zero_pdf)
                })
            };
            match self.mode {
                BsdfBlendMode::Addictive => {
                    sample.color = sample.color + f_a;
                },
                BsdfBlendMode::Mix => {
                    sample.color = f_a * (1.0 - frac) + sample.color * frac;
                }
            }
            sample.pdf = pdf_a * (1.0 - frac) + sample.pdf * frac;
            sample
        })
    }

    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Float {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        let zero = const_(0.0f32);
        let pdf_a = if_!(frac.cmplt(1.0 - Self::EPS), {
            self.bsdf_a.pdf(wo, wi,ctx)
        }, else {
            zero
        });
        let pdf_b = if_!(frac.cmpgt(Self::EPS), {
            self.bsdf_b.pdf(wo, wi,ctx)
        }, else {
            zero
        });
        pdf_a * (1.0 - frac) + pdf_b * frac
    }
    fn albedo(&self, wo: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        match self.mode {
            BsdfBlendMode::Addictive => self.bsdf_a.albedo(wo, ctx) + self.bsdf_b.albedo(wo, ctx),
            BsdfBlendMode::Mix => {
                self.bsdf_a.albedo(wo, ctx) * (1.0 - frac) + self.bsdf_b.albedo(wo, ctx) * frac
            }
        }
    }
}

pub struct BsdfClosure {
    inner: Box<dyn Bsdf>,
    frame: Expr<Frame>,
}

impl Bsdf for BsdfClosure {
    // return f(wo, wi) * abs_cos_theta(wi)
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        self.inner
            .evaluate(self.frame.to_local(wo), self.frame.to_local(wi), ctx)
    }

    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let sample = self
            .inner
            .sample(self.frame.to_local(wo), u_select, u_sample, ctx);
        BsdfSample {
            wi: self.frame.to_world(sample.wi),
            ..sample
        }
    }

    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Float {
        self.inner
            .pdf(self.frame.to_local(wo), self.frame.to_local(wi), ctx)
    }
    fn albedo(&self, wo: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        self.inner.albedo(self.frame.to_local(wo), ctx)
    }
}
pub trait Fresnel {
    fn evaluate(&self, cos_theta_i: Expr<f32>, ctx: &BsdfEvalContext) -> Color;
}
pub struct MicrofacetReflection {
    pub color: Color,
    pub fresnel: Box<dyn Fresnel>,
    pub dist: Box<dyn MicrofacetDistribution>,
}

impl Bsdf for MicrofacetReflection {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        let wh = wo + wi;
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        if_!((wh.dot(wo) * wi.dot(wh)).cmplt(0.0)
            | wh.cmpeq(0.0).all()
            | cos_i.cmpeq(0.0)
            | cos_o.cmpeq(0.0)
            | !Frame::same_hemisphere(wo, wi), {
                Color::zero(ctx.color_repr)
        }, else {
            let wh = wh.normalize();
            let f = self.fresnel.evaluate(wi.dot(face_forward(wh, make_float3(0.0,1.0,0.0))), ctx);
            let d = self.dist.d(wh);
            let g = self.dist.g(wo, wi);
            &self.color * &f * (0.25 * d * g / (cos_i * cos_o)).abs() * cos_o.abs()
        })
    }

    fn sample(
        &self,
        wo: Expr<Float3>,
        _u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let wh = self.dist.sample_wh(wo, u_sample);
        let wi = reflect(wo, wh);
        let valid = Frame::same_hemisphere(wo, wi);
        BsdfSample {
            color: self.evaluate(wo, wi, ctx),
            pdf: self.pdf(wo, wi, ctx),
            valid,
            wi,
            lobe_roughness: self.dist.roughness(),
        }
    }

    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, _ctx: &BsdfEvalContext) -> Float {
        let wh = wo + wi;
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        if_!((wh.dot(wo) * wi.dot(wh)).cmplt(0.0)
            | wh.cmpeq(0.0).all()
            | cos_i.cmpeq(0.0)
            | cos_o.cmpeq(0.0)
            | !Frame::same_hemisphere(wo, wi), {
                const_(0.0f32)
        }, else {
            let wh = wh.normalize();
            self.dist.pdf(wo, wh) / (4.0 * wo.dot(wh))
        })
    }
    fn albedo(&self, _wo: Expr<Float3>, _ctx: &BsdfEvalContext) -> Color {
        self.color
    }
}

pub struct MicrofacetTransmission {
    pub dist: Box<dyn MicrofacetDistribution>,
    pub color: Color,
    pub eta: Expr<f32>,
    pub fresnel: Box<dyn Fresnel>,
}

impl Bsdf for MicrofacetTransmission {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(cos_o.cmpgt(0.0), self.eta, 1.0 / self.eta);
        let wh = (wo + wi * eta).normalize();
        let wh = face_forward(wh, make_float3(0.0, 1.0, 0.0));
        if_!((wh.dot(wo) * wi.dot(wh)).cmpgt(0.0)
            | cos_i.cmpeq(0.0)
            | cos_o.cmpeq(0.0)
            | Frame::same_hemisphere(wo, wi), {
            Color::zero(ctx.color_repr)
        }, else {
            let f = self.fresnel.evaluate(wo.dot(wh), ctx);
            let denom = (wi.dot(wh)  + wo.dot(wh) / eta).sqr() * cos_i * cos_o;
            (Color::one(ctx.color_repr) - f)
                * &self.color *(self.dist.d(wh)
                * self.dist.g(wo, wi) * eta.sqr()
                * wi.dot(wh).abs() * wo.dot(wh).abs()
                / denom).abs() * cos_o.abs()

            // Float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap) * cosTheta_i * cosTheta_o;
            // Float ft = mfDistrib.D(wm) * (1 - F) * mfDistrib.G(wo, wi) *
            //            std::abs(Dot(wi, wm) * Dot(wo, wm) / denom);
            // // Account for non-symmetry with transmission to different medium
            // if (mode == TransportMode::Radiance)
            //     ft /= Sqr(etap);

            // return SampledSpectrum(ft);
        })
    }

    fn sample(
        &self,
        wo: Expr<Float3>,
        _u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let wh = self.dist.sample_wh(wo, u_sample);
        let (refracted, _eta, wi) = refract(wo, wh, self.eta);
        let valid = refracted & !Frame::same_hemisphere(wo, wi);
        let pdf = self.pdf(wo, wi, ctx);
        // lc_assert!(pdf.cmpgt(0.0) | !valid);
        BsdfSample {
            pdf,
            color: self.evaluate(wo, wi, ctx),
            wi,
            valid,
            lobe_roughness: self.dist.roughness(),
        }
    }

    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, _ctx: &BsdfEvalContext) -> Float {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(cos_o.cmpgt(0.0), self.eta, 1.0 / self.eta);
        let wh = (wo + wi * eta).normalize();
        let wh = face_forward(wh, make_float3(0.0, 1.0, 0.0));
        if_!((wh.dot(wo) * wi.dot(wh)).cmpgt(0.0)
            | cos_i.cmpeq(0.0)
            | cos_o.cmpeq(0.0)
            | Frame::same_hemisphere(wo, wi), {
            const_(0.0f32)
        }, else {
            // Float denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            // Float dwm_dwi = AbsDot(wi, wm) / denom;
            // pdf = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
            let denom = (wi.dot(wh) + wo.dot(wh) /  eta).sqr();
            let dwh_dwi = wi.dot(wh).abs() / denom;
            self.dist.pdf(wo, wh) * dwh_dwi
        })
    }
    fn albedo(&self, _wo: Expr<Float3>, _ctx: &BsdfEvalContext) -> Color {
        self.color
    }
}

pub fn fr_dielectric(cos_theta_i: Expr<f32>, eta: Expr<f32>) -> Expr<f32> {
    let cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);
    let eta = select(cos_theta_i.cmpgt(0.0), eta, 1.0 / eta);
    let cos_theta_i = cos_theta_i.abs();
    //
    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    let sin2_theta_i = 1.0 - cos_theta_i.sqr();
    let sin2_theta_t = sin2_theta_i / eta.sqr();
    if_!(sin2_theta_t.cmpge(1.0), {
        const_(1.0f32)
    }, else {
        let cos_theta_t = (1.0 - sin2_theta_t).max(0.0).sqrt();
        let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
        let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
        let fr = (r_parl.sqr() + r_perp.sqr()) * 0.5;
        // cpu_dbg!(fr);
        fr.clamp(0.0, 1.0)
    })
    //     Float sin2Theta_i = 1 - Sqr(cosTheta_i);
    //     Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    //     if (sin2Theta_t >= 1)
    //         return 1.f;
    //     Float cosTheta_t = SafeSqrt(1 - sin2Theta_t);

    //     Float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    //     Float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    //     return (Sqr(r_parl) + Sqr(r_perp)) / 2;
}

pub fn fr_schlick(f0: Color, cos_theta_i: Expr<f32>) -> Color {
    let cos_theta_i = cos_theta_i.clamp(-1.0, 1.0).abs();
    let pow5 = |x: Expr<f32>| x.sqr().sqr() * x;
    let fr = f0 + (Color::one(f0.repr()) - f0) * pow5(1.0 - cos_theta_i).abs();
    fr
}
#[derive(Copy, Clone)]
pub struct FresnelSchlick {
    pub f0: Color,
}
impl Fresnel for FresnelSchlick {
    fn evaluate(&self, cos_theta_i: Expr<f32>, _ctx: &BsdfEvalContext) -> Color {
        fr_schlick(self.f0, cos_theta_i)
    }
}
#[derive(Copy, Clone)]
pub struct FresnelDielectric {
    pub eta: Expr<f32>, //eta = eta_t / eta_i
}
impl Fresnel for FresnelDielectric {
    fn evaluate(&self, cos_theta_i: Expr<f32>, ctx: &BsdfEvalContext) -> Color {
        Color::one(ctx.color_repr) * fr_dielectric(cos_theta_i, self.eta)
    }
}
#[derive(Copy, Clone)]
pub struct ConstFresnel {}
impl Fresnel for ConstFresnel {
    fn evaluate(&self, _: Expr<f32>, ctx: &BsdfEvalContext) -> Color {
        Color::one(ctx.color_repr)
    }
}

#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct FlatBsdfSample {
    pub wi: Float3,
    pub pdf: f32,
    pub color: FlatColor,
    pub valid: bool,
    pub lobe_roughness: f32,
}
pub const BSDF_EVAL_COLOR: u32 = 1 << 0;
pub const BSDF_EVAL_PDF: u32 = 1 << 1;
pub const BSDF_EVAL_LOBE_ROUGHNESS: u32 = 1 << 2;

#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct FlatBsdfEvalResult {
    pub color: FlatColor,
    pub pdf: f32,
    pub lobe_roughness: f32,
}
pub struct BsdfEvaluator {
    pub(crate) color_repr: ColorRepr,
    // bsdf(surface, si, wo, wi, u_select, flags)
    pub(crate) bsdf: Callable<
        (
            Expr<TagIndex>,
            Expr<SurfaceInteraction>,
            Expr<Float3>,
            Expr<Float3>,
            Expr<f32>,
            Expr<u32>,
        ),
        Expr<FlatBsdfEvalResult>,
    >,
    // bsdf_sample(surface, si, wo, (u_select, u_sample))
    pub(crate) bsdf_sample: Callable<
        (
            Expr<TagIndex>,
            Expr<SurfaceInteraction>,
            Expr<Float3>,
            Expr<Float3>,
        ),
        Expr<FlatBsdfSample>,
    >,
    pub(crate) albedo:
        Callable<(Expr<TagIndex>, Expr<SurfaceInteraction>, Expr<Float3>), Expr<FlatColor>>,
}
impl BsdfEvaluator {
    pub fn evaluate(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
    ) -> Color {
        let result = self
            .bsdf
            .call(surface, si, wo, wi, const_(0.0f32), const_(BSDF_EVAL_COLOR));
        Color::from_flat(self.color_repr, result.color())
    }
    pub fn pdf(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
    ) -> Float {
        let result = self
            .bsdf
            .call(surface, si, wo, wi, const_(0.0f32), const_(BSDF_EVAL_PDF));
        result.pdf()
    }
    pub fn evaluate_color_and_pdf(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
    ) -> (Color, Float) {
        let result = self.bsdf.call(
            surface,
            si,
            wo,
            wi,
            const_(0.0f32),
            const_(BSDF_EVAL_COLOR | BSDF_EVAL_PDF),
        );
        (
            Color::from_flat(self.color_repr, result.color()),
            result.pdf(),
        )
    }
    pub fn albedo(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
    ) -> Color {
        let result = self.albedo.call(surface, si, wo);
        Color::from_flat(self.color_repr, result)
    }
    pub fn sample(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        u: Expr<Float3>,
    ) -> BsdfSample {
        let sample = self.bsdf_sample.call(surface, si, wo, u);
        BsdfSample {
            wi: sample.wi(),
            pdf: sample.pdf(),
            color: Color::from_flat(self.color_repr, sample.color()),
            valid: sample.valid(),
            lobe_roughness: sample.lobe_roughness(),
        }
    }
}