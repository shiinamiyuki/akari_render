use std::rc::Rc;

use crate::color::{ColorRepr, FlatColor, SampledWavelengths};
use crate::geometry::{face_forward, reflect, refract};
use crate::microfacet::MicrofacetDistribution;
use crate::sampling::weighted_discrete_choice2_and_remap;
use crate::svm::eval::SvmEvaluator;
use crate::{color::Color, geometry::Frame, interaction::SurfaceInteraction, *};

use super::ShaderRef;

pub struct BsdfEvalContext<'a> {
    pub color_repr: ColorRepr,
    pub ad_mode: Option<ADMode>,
    pub _marker: std::marker::PhantomData<&'a ()>,
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
}

pub trait Surface {
    // return f(wo, wi) * abs_cos_theta(wi)
    fn evaluate(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color;
    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample;
    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Float;
    fn albedo(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color;
    fn roughness(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32>;
    fn emission(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color;
}

pub trait SurfaceShader {
    fn closure(&self, svm_eval: &SvmEvaluator<'_>) -> Rc<dyn Surface>;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BsdfBlendMode {
    Addictive,
    Mix,
}
pub struct EmissiveSurface {
    pub inner: Option<Rc<dyn Surface>>,
    pub emission: Color,
}
impl Surface for EmissiveSurface {
    fn evaluate(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        if let Some(inner) = &self.inner {
            inner.evaluate(wo, wi, swl, ctx)
        } else {
            Color::zero(ctx.color_repr)
        }
    }

    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        if let Some(inner) = &self.inner {
            inner.sample(wo, u_select, u_sample, swl, ctx)
        } else {
            BsdfSample {
                wi: make_float3(0.0, 0.0, 0.0),
                pdf: const_(0.0f32),
                color: Color::zero(ctx.color_repr),
                valid: const_(false),
            }
        }
    }

    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Float {
        if let Some(inner) = &self.inner {
            inner.pdf(wo, wi, swl, ctx)
        } else {
            const_(0.0f32)
        }
    }

    fn albedo(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        if let Some(inner) = &self.inner {
            inner.albedo(wo, swl, ctx)
        } else {
            Color::zero(ctx.color_repr)
        }
    }

    fn roughness(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        if let Some(inner) = &self.inner {
            inner.roughness(wo, swl, ctx)
        } else {
            const_(1.0f32)
        }
    }

    fn emission(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        if let Some(inner) = &self.inner {
            self.emission + inner.emission(wo, swl, ctx)
        } else {
            self.emission
        }
    }
}
pub struct BsdfMixture {
    pub frac: Box<dyn Fn(Expr<Float3>, &BsdfEvalContext) -> Expr<f32>>,
    pub bsdf_a: Rc<dyn Surface>,
    pub bsdf_b: Rc<dyn Surface>,
    pub mode: BsdfBlendMode,
}
impl BsdfMixture {
    const EPS: f32 = 1e-4;
}

impl Surface for BsdfMixture {
    fn evaluate(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        match self.mode {
            BsdfBlendMode::Addictive => {
                let f_a = self.bsdf_a.evaluate(wo, wi, swl, ctx);
                let f_b = self.bsdf_b.evaluate(wo, wi, swl, ctx);
                f_a + f_b
            }
            BsdfBlendMode::Mix => {
                let frac: Expr<f32> = (self.frac)(wo, ctx);
                let zero = Color::zero(ctx.color_repr);
                let f_a = if_!(
                    frac.cmplt(1.0 - Self::EPS),
                    { self.bsdf_a.evaluate(wo, wi, swl, ctx) },
                    else,
                    { zero }
                );
                let f_b = if_!(
                    frac.cmpgt(Self::EPS),
                    { self.bsdf_b.evaluate(wo, wi, swl, ctx) },
                    else,
                    { zero }
                );
                f_a * (1.0 - frac) + f_b * frac
            }
        }
    }

    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        let (which, remapped) =
            weighted_discrete_choice2_and_remap(frac, const_(1u32), const_(0u32), u_select);
        let zero_f = Color::zero(ctx.color_repr);
        let zero_pdf = const_(0.0f32);
        if_!(
            which.cmpeq(0),
            {
                let mut sample = self.bsdf_a.sample(wo, remapped, u_sample, swl, ctx);
                let (f_b, pdf_b) = {
                    if_!(
                        frac.cmpgt(Self::EPS),
                        {
                            let f_b = self.bsdf_b.evaluate(wo, sample.wi, *swl, ctx);
                            let pdf_b = self.bsdf_b.pdf(wo, sample.wi, *swl, ctx);
                            (f_b, pdf_b)
                        },
                        else,
                        { (zero_f, zero_pdf) }
                    )
                };
                // cpu_dbg!(make_float2(sample.pdf, pdf_b));
                match self.mode {
                    BsdfBlendMode::Addictive => {
                        sample.color = sample.color + f_b;
                    }
                    BsdfBlendMode::Mix => {
                        sample.color = sample.color * (1.0 - frac) + f_b * frac;
                    }
                }

                sample.pdf = sample.pdf * (1.0 - frac) + pdf_b * frac;
                sample
            },
            else,
            {
                let mut sample = self.bsdf_b.sample(wo, remapped, u_sample, swl, ctx);
                let (f_a, pdf_a) = {
                    if_!(
                        frac.cmplt(1.0 - Self::EPS),
                        {
                            let f_a = self.bsdf_a.evaluate(wo, sample.wi, *swl, ctx);
                            let pdf_a = self.bsdf_a.pdf(wo, sample.wi, *swl, ctx);
                            (f_a, pdf_a)
                        },
                        else,
                        { (zero_f, zero_pdf) }
                    )
                };
                match self.mode {
                    BsdfBlendMode::Addictive => {
                        sample.color = sample.color + f_a;
                    }
                    BsdfBlendMode::Mix => {
                        sample.color = f_a * (1.0 - frac) + sample.color * frac;
                    }
                }
                sample.pdf = pdf_a * (1.0 - frac) + sample.pdf * frac;
                sample
            }
        )
    }

    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Float {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        let zero = const_(0.0f32);
        let pdf_a = if_!(
            frac.cmplt(1.0 - Self::EPS),
            { self.bsdf_a.pdf(wo, wi, swl, ctx) },
            else,
            { zero }
        );
        let pdf_b = if_!(
            frac.cmpgt(Self::EPS),
            { self.bsdf_b.pdf(wo, wi, swl, ctx) },
            else,
            { zero }
        );
        pdf_a * (1.0 - frac) + pdf_b * frac
    }
    fn albedo(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        match self.mode {
            BsdfBlendMode::Addictive => {
                self.bsdf_a.albedo(wo, swl, ctx) + self.bsdf_b.albedo(wo, swl, ctx)
            }
            BsdfBlendMode::Mix => {
                self.bsdf_a.albedo(wo, swl, ctx) * (1.0 - frac)
                    + self.bsdf_b.albedo(wo, swl, ctx) * frac
            }
        }
    }
    fn roughness(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        self.bsdf_a.roughness(wo, swl, ctx) * (1.0 - frac)
            + self.bsdf_b.roughness(wo, swl, ctx) * frac
    }
    fn emission(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        match self.mode {
            BsdfBlendMode::Addictive => {
                self.bsdf_a.emission(wo, swl, ctx) + self.bsdf_b.emission(wo, swl, ctx)
            }
            BsdfBlendMode::Mix => {
                self.bsdf_a.emission(wo, swl, ctx) * (1.0 - frac)
                    + self.bsdf_b.emission(wo, swl, ctx) * frac
            }
        }
    }
}

pub struct SurfaceClosure {
    pub inner: Rc<dyn Surface>,
    pub frame: Expr<Frame>,
}

impl Surface for SurfaceClosure {
    // return f(wo, wi) * abs_cos_theta(wi)
    fn evaluate(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner
            .evaluate(self.frame.to_local(wo), self.frame.to_local(wi), swl, ctx)
    }

    fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let sample = self
            .inner
            .sample(self.frame.to_local(wo), u_select, u_sample, swl, ctx);
        BsdfSample {
            wi: self.frame.to_world(sample.wi),
            ..sample
        }
    }

    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Float {
        self.inner
            .pdf(self.frame.to_local(wo), self.frame.to_local(wi), swl, ctx)
    }
    fn albedo(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.albedo(self.frame.to_local(wo), swl, ctx)
    }
    fn roughness(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.inner.roughness(self.frame.to_local(wo), swl, ctx)
    }
    fn emission(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.emission(self.frame.to_local(wo), swl, ctx)
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

impl Surface for MicrofacetReflection {
    fn evaluate(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
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
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let wh = self.dist.sample_wh(wo, u_sample);
        let wi = reflect(wo, wh);
        let valid = Frame::same_hemisphere(wo, wi);
        BsdfSample {
            color: self.evaluate(wo, wi, *swl, ctx),
            pdf: self.pdf(wo, wi, *swl, ctx),
            valid,
            wi,
        }
    }

    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Float {
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
    fn albedo(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.color
    }
    fn roughness(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.dist.roughness()
    }
    fn emission(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        Color::zero(ctx.color_repr)
    }
}

pub struct MicrofacetTransmission {
    pub dist: Box<dyn MicrofacetDistribution>,
    pub color: Color,
    pub eta: Expr<f32>,
    pub fresnel: Box<dyn Fresnel>,
}

impl Surface for MicrofacetTransmission {
    fn evaluate(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(cos_o.cmpgt(0.0), self.eta, 1.0 / self.eta);
        let wh = (wo + wi * eta).normalize();
        let wh = face_forward(wh, make_float3(0.0, 1.0, 0.0));
        let backfacing = (wh.dot(wi) * cos_i).cmplt(0.0) | (wh.dot(wo) * cos_o).cmplt(0.0);
        if_!((wh.dot(wo) * wi.dot(wh)).cmpgt(0.0)
            | cos_i.cmpeq(0.0)
            | cos_o.cmpeq(0.0)
            | backfacing
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
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let wh = self.dist.sample_wh(wo, u_sample);
        let (refracted, _eta, wi) = refract(wo, wh, self.eta);
        let valid = refracted & !Frame::same_hemisphere(wo, wi);
        let pdf = self.pdf(wo, wi, *swl, ctx);
        let valid = valid & pdf.cmpgt(0.0);
        // lc_assert!(pdf.cmpgt(0.0) | !valid);
        BsdfSample {
            pdf,
            color: self.evaluate(wo, wi, *swl, ctx),
            wi,
            valid,
        }
    }

    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Float {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(cos_o.cmpgt(0.0), self.eta, 1.0 / self.eta);
        let wh = (wo + wi * eta).normalize();
        let wh = face_forward(wh, make_float3(0.0, 1.0, 0.0));
        let backfacing = (wh.dot(wi) * cos_i).cmplt(0.0) | (wh.dot(wo) * cos_o).cmplt(0.0);
        if_!((wh.dot(wo) * wi.dot(wh)).cmpgt(0.0)
            | cos_i.cmpeq(0.0)
            | cos_o.cmpeq(0.0)
            | backfacing
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
    fn albedo(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.color
    }
    fn roughness(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.dist.roughness()
    }
    fn emission(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        Color::zero(ctx.color_repr)
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
}
pub const SURFACE_EVAL_COLOR: u32 = 1 << 0;
pub const SURFACE_EVAL_PDF: u32 = 1 << 1;
pub const SURFACE_EVAL_ROUGHNESS: u32 = 1 << 2;
pub const SURFACE_EVAL_ALBEDO: u32 = 1 << 3;
pub const SURFACE_EVAL_EMISSION: u32 = 1 << 4;
pub const SURFACE_EVAL_ALL: u32 = SURFACE_EVAL_COLOR
    | SURFACE_EVAL_PDF
    | SURFACE_EVAL_ROUGHNESS
    | SURFACE_EVAL_ALBEDO
    | SURFACE_EVAL_EMISSION;

#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct FlatSurfaceEvalResult {
    pub color: FlatColor,
    pub pdf: f32,
    pub albedo: FlatColor,
    pub emission: FlatColor,
    pub roughness: f32,
}
pub struct SurfaceEvaluator {
    pub(crate) color_repr: ColorRepr,
    // bsdf(surface, si, wo, wi, u_select, flags)
    pub(crate) eval: Callable<
        fn(
            Expr<ShaderRef>,
            Expr<SurfaceInteraction>,
            Expr<Float3>,
            Expr<Float3>,
            Expr<SampledWavelengths>,
            Expr<u32>,
        ) -> Expr<FlatSurfaceEvalResult>,
    >,
    // bsdf_sample(surface, si, wo, (u_select, u_sample))
    pub(crate) sample: Callable<
        fn(
            Expr<ShaderRef>,
            Expr<SurfaceInteraction>,
            Expr<Float3>,
            Expr<Float3>,
            Var<SampledWavelengths>,
        ) -> Expr<FlatBsdfSample>,
    >,
}
impl SurfaceEvaluator {
    pub fn evaluate_ex(
        &self,
        surface: Expr<ShaderRef>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        flags: Expr<u32>,
    ) -> Expr<FlatSurfaceEvalResult> {
        self.eval.call(surface, si, wo, wi, swl, flags)
    }
    pub fn evaluate(
        &self,
        surface: Expr<ShaderRef>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
    ) -> Color {
        let result = self
            .eval
            .call(surface, si, wo, wi, swl, const_(SURFACE_EVAL_COLOR));
        Color::from_flat(self.color_repr, result.color())
    }
    pub fn pdf(
        &self,
        surface: Expr<ShaderRef>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
    ) -> Float {
        let result = self
            .eval
            .call(surface, si, wo, wi, swl, const_(SURFACE_EVAL_PDF));
        result.pdf()
    }
    pub fn evaluate_color_and_pdf(
        &self,
        surface: Expr<ShaderRef>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
    ) -> (Color, Float) {
        let result = self.eval.call(
            surface,
            si,
            wo,
            wi,
            swl,
            const_(SURFACE_EVAL_COLOR | SURFACE_EVAL_PDF),
        );
        (
            Color::from_flat(self.color_repr, result.color()),
            result.pdf(),
        )
    }
    pub fn albedo(
        &self,
        surface: Expr<ShaderRef>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
    ) -> Color {
        let result = self.eval.call(
            surface,
            si,
            wo,
            Float3Expr::zero(),
            swl,
            const_(SURFACE_EVAL_ALBEDO),
        );
        Color::from_flat(self.color_repr, result.albedo())
    }
    pub fn sample(
        &self,
        surface: Expr<ShaderRef>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        u: Expr<Float3>,
        swl: Var<SampledWavelengths>,
    ) -> BsdfSample {
        let sample = self.sample.call(surface, si, wo, u, swl);
        BsdfSample {
            wi: sample.wi(),
            pdf: sample.pdf(),
            color: Color::from_flat(self.color_repr, sample.color()),
            valid: sample.valid(),
        }
    }
}
