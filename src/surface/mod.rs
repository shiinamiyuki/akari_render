use crate::color::ColorRepr;
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
}

pub trait Surface {
    fn closure(&self, si: Expr<SurfaceInteraction>, ctx: &BsdfEvalContext) -> BsdfClosure;
}

pub struct BsdfMixture {
    pub frac: Box<dyn Fn(Expr<Float3>, &BsdfEvalContext) -> Expr<f32>>,
    pub bsdf_a: Box<dyn Bsdf>,
    pub bsdf_b: Box<dyn Bsdf>,
}

impl Bsdf for BsdfMixture {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        let f_a = self.bsdf_a.evaluate(wo, wi, ctx);
        let f_b = self.bsdf_b.evaluate(wo, wi, ctx);
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        f_a * (1.0 - frac) + f_b * frac
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
            weighted_discrete_choice2_and_remap(frac, const_(0u32), const_(1u32), u_select);
        if_!(which.cmpeq(0), {
            let mut sample = self.bsdf_a.sample(wo, remapped,u_sample, ctx);
            let f_b = self.bsdf_b.evaluate(wo, sample.wi, ctx);
            let pdf_b = self.bsdf_b.pdf(wo, sample.wi, ctx);
            sample.color = sample.color * (1.0 - frac) + f_b * frac;
            sample.pdf = sample.pdf * (1.0 - frac) + pdf_b * frac;
            sample
        }, else {
            let mut sample = self.bsdf_b.sample(wo, remapped, u_sample, ctx);
            let f_a = self.bsdf_a.evaluate(wo, sample.wi, ctx);
            let pdf_a = self.bsdf_a.pdf(wo, sample.wi, ctx);
            sample.color = f_a * (1.0 - frac) + sample.color * frac;
            sample.pdf = pdf_a * (1.0 - frac) + sample.pdf * frac;
            sample
        })
    }

    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Float {
        let pdf_a = self.bsdf_a.pdf(wo, wi, ctx);
        let pdf_b = self.bsdf_b.pdf(wo, wi, ctx);
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        pdf_a * (1.0 - frac) + pdf_b * frac
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
        if_!(Frame::same_hemisphere(wo, wi), {
            let wh = wh.normalize();
            let f = self.fresnel.evaluate(wi.dot(face_forward(wh, make_float3(0.0,1.0,0.0))), ctx);
            let d = self.dist.d(wh);
            let g = self.dist.g(wo, wi);
            let cos_o = Frame::cos_theta(wo);
            let cos_i = Frame::cos_theta(wi);
            &self.color * &f * (0.25 * d * g / (cos_i * cos_o)).abs() * cos_o.abs()
        }, else {
            Color::zero(ctx.color_repr)
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
        }
    }

    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, _ctx: &BsdfEvalContext) -> Float {
        let wh = wo + wi;
        if_!(Frame::same_hemisphere(wo, wi) & wh.cmpne(0.0).any(), {
            let wh = wh.normalize();
            self.dist.pdf(wo, wh) / (4.0 * wo.dot(wh))
        }, else {
            const_(0.0f32)
        })
    }
}

pub struct MicrofacetTransmission {
    pub dist: Box<dyn MicrofacetDistribution>,
    pub color: Color,
    pub eta_a: Expr<f32>,
    pub eta_b: Expr<f32>,
    pub fresnel: Box<dyn Fresnel>,
}

impl Bsdf for MicrofacetTransmission {
    fn evaluate(&self, wo: Expr<Float3>, wi: Expr<Float3>, ctx: &BsdfEvalContext) -> Color {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(
            cos_o.cmpgt(0.0),
            self.eta_b / self.eta_a,
            self.eta_a / self.eta_b,
        );
        let wh = (wo + wi * eta).normalize();
        let wh = select(wh.y().cmplt(0.0), -wh, wh);
        if_!((wh.dot(wo) * wi.dot(wh)).cmpgt(0.0)
            | cos_i.cmpeq(0.0)
            | cos_o.cmpeq(0.0)
            | Frame::same_hemisphere(wo, wi), {
            Color::zero(ctx.color_repr)
        }, else {
            let f = self.fresnel.evaluate(wo.dot(wh), ctx);
            let sqrt_denom = wo.dot(wh) + eta * wi.dot(wh);
            (Color::one(ctx.color_repr) - f)
                * &self.color *(self.dist.d(wh)
                * self.dist.g(wo, wi) * eta.sqr()
                * wi.dot(wh).abs() * wo.dot(wh).abs()
                / (cos_i * cos_o * sqrt_denom.sqr())).abs() * cos_o.abs()
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
        let cos_o = Frame::cos_theta(wo);
        let eta = select(
            cos_o.cmpgt(0.0),
            self.eta_a / self.eta_b,
            self.eta_b / self.eta_a,
        );
        let valid = wo.dot(wh).cmpgt(0.0);
        let (refracted, wi) = refract(wo, wh, eta);
        let valid = refracted & valid;
        BsdfSample {
            pdf: self.pdf(wo, wi, ctx),
            color: self.evaluate(wo, wi, ctx),
            wi,
            valid,
        }
    }

    fn pdf(&self, wo: Expr<Float3>, wi: Expr<Float3>, _ctx: &BsdfEvalContext) -> Float {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(
            cos_o.cmpgt(0.0),
            self.eta_b / self.eta_a,
            self.eta_a / self.eta_b,
        );
        let wh = (wo + wi * eta).normalize();
        if_!((wh.dot(wo) * wi.dot(wh)).cmpgt(0.0)
            | cos_i.cmpeq(0.0)
            | cos_o.cmpeq(0.0)
            | Frame::same_hemisphere(wo, wi), {
            const_(0.0f32)
        }, else {
            let sqrt_denom =wo.dot(wh) + eta * wi.dot(wh);
            let dwh_dwi = (eta * eta * wi.dot(wh) / sqrt_denom.sqr()).abs();
            self.dist.pdf(wo, wh) * dwh_dwi
        })
    }
}

pub fn fr_dielectric(cos_theta_i: Expr<f32>, eta_i: Expr<f32>, eta_t: Expr<f32>) -> Expr<f32> {
    let cos_theta_i = cos_theta_i.clamp(-1.0, 1.0);

    let entering = cos_theta_i.cmpgt(0.0);
    let (eta_i, eta_t) = if_!(!entering, {
        (eta_t, eta_i)
    }, else {
        (eta_i, eta_t)
    });
    let cos_theta_i = cos_theta_i.abs();
    let sin_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0).sqrt();
    let sin_theta_t = eta_i / eta_t * sin_theta_i;

    let sin_theta_t = sin_theta_t.min(1.0);
    let cos_theta_t = (1.0 - sin_theta_t * sin_theta_t).max(0.0).sqrt();
    let rparl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t))
        / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    let rperp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t))
        / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    (rparl * rparl + rperp * rperp) / 2.0
}
#[derive(Copy, Clone)]
pub struct FresnelDielectric {
    pub eta_i: Expr<f32>,
    pub eta_t: Expr<f32>,
}
impl Fresnel for FresnelDielectric {
    fn evaluate(&self, cos_theta_i: Expr<f32>, ctx: &BsdfEvalContext) -> Color {
        Color::one(ctx.color_repr) * fr_dielectric(cos_theta_i, self.eta_i, self.eta_t)
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
pub struct TBsdfSample<T: Value> {
    pub wi: Float3,
    pub pdf: f32,
    pub color: T,
    pub valid: bool,
}
pub const BSDF_EVAL_COLOR: u32 = 1 << 0;
pub const BSDF_EVAL_PDF: u32 = 1 << 1;
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct BsdfEvalResult<T: Value> {
    pub color: T,
    pub pdf: f32,
}
pub struct BsdfEvaluator {
    pub(crate) color_repr: ColorRepr,
    pub(crate) bsdf: DynCallable<
        (
            Expr<TagIndex>,
            Expr<SurfaceInteraction>,
            Expr<Float3>,
            Expr<Float3>,
            Expr<u32>,
        ),
        DynExpr,
    >,
    pub(crate) bsdf_sample: DynCallable<
        (
            Expr<TagIndex>,
            Expr<SurfaceInteraction>,
            Expr<Float3>,
            Expr<Float3>,
        ),
        DynExpr,
    >,
}
impl BsdfEvaluator {
    pub fn evaluate(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
    ) -> Color {
        let result = self.bsdf.call(surface, si, wo, wi, const_(BSDF_EVAL_COLOR));
        match self.color_repr {
            ColorRepr::Rgb => Color::Rgb(result.get::<BsdfEvalResult<Float3>>().color()),
            _ => todo!(),
        }
    }
    pub fn pdf(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
    ) -> Float {
        let result = self.bsdf.call(surface, si, wo, wi, const_(BSDF_EVAL_COLOR));
        match self.color_repr {
            ColorRepr::Rgb => result.get::<BsdfEvalResult<Float3>>().pdf(),
            _ => todo!(),
        }
    }
    pub fn evaluate_color_and_pdf(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
    ) -> (Color, Float) {
        let result = self.bsdf.call(surface, si, wo, wi, const_(BSDF_EVAL_COLOR));
        match self.color_repr {
            ColorRepr::Rgb => {
                let result = result.get::<BsdfEvalResult<Float3>>();
                (Color::Rgb(result.color()), result.pdf())
            }
            _ => todo!(),
        }
    }
    pub fn sample(
        &self,
        surface: Expr<TagIndex>,
        si: Expr<SurfaceInteraction>,
        wo: Expr<Float3>,
        u: Expr<Float3>,
    ) -> BsdfSample {
        let sample = self.bsdf_sample.call(surface, si, wo, u);
        match self.color_repr {
            ColorRepr::Rgb => {
                let sample = sample.get::<TBsdfSample<Float3>>();
                BsdfSample {
                    wi: sample.wi(),
                    pdf: sample.pdf(),
                    color: Color::Rgb(sample.color()),
                    valid: sample.valid(),
                }
            }
            _ => todo!(),
        }
    }
}
