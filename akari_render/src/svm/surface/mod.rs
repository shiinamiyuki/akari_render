use std::rc::Rc;

use crate::color::{ColorRepr, ColorVar, FlatColor, SampledWavelengths};
use crate::geometry::{face_forward, reflect, refract};
use crate::microfacet::MicrofacetDistribution;
use crate::sampling::weighted_discrete_choice2_and_remap;
use crate::svm::eval::SvmEvaluator;
use crate::{color::Color, geometry::Frame, interaction::SurfaceInteraction, *};

#[derive(Clone, Copy, Debug)]
pub struct BsdfEvalContext {
    pub color_repr: ColorRepr,
    pub ad_mode: ADMode,
}

pub mod diffuse;
pub mod glass;
pub mod principled;

#[derive(Clone, Aggregate)]
pub struct BsdfSample {
    pub wi: Expr<Float3>,
    pub pdf: Expr<f32>,
    pub color: Color,
    pub valid: Expr<bool>,
}

pub trait Surface {
    // return f(wo, wi) * abs_cos_theta(wi)
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color;
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>);
    fn pdf_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32>;
    fn albedo_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color;
    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32>;
    fn emission_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color;

    /// wrappers

    fn evaluate(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let ret = ColorVar::zero(ctx.color_repr);
        maybe_outline(|| ret.store(self.evaluate_impl(wo, wi, swl, ctx)));
        ret.load()
    }
    fn sample_wi(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let wi = Var::<Float3>::zeroed();
        let valid = Var::<bool>::zeroed();
        maybe_outline(|| {
            let (wi_, valid_) = self.sample_wi_impl(wo, u_select, u_sample, swl, ctx);
            wi.store(wi_);
            valid.store(valid_);
        });
        (wi.load(), valid.load())
    }
    fn pdf(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let ret = Var::<f32>::zeroed();
        maybe_outline(|| ret.store(self.pdf_impl(wo, wi, swl, ctx)));
        ret.load()
    }
    fn albedo(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let ret = ColorVar::zero(ctx.color_repr);
        maybe_outline(|| ret.store(self.albedo_impl(wo, swl, ctx)));
        ret.load()
    }
    fn roughness(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let ret = Var::<f32>::zeroed();
        maybe_outline(|| ret.store(self.roughness_impl(wo, swl, ctx)));
        ret.load()
    }
    fn emission(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let ret = ColorVar::zero(ctx.color_repr);
        maybe_outline(|| ret.store(self.emission_impl(wo, swl, ctx)));
        ret.load()
    }
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
    fn evaluate_impl(
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

    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        if let Some(inner) = &self.inner {
            inner.sample_wi(wo, u_select, u_sample, swl, ctx)
        } else {
            (Expr::<Float3>::zeroed(), false.expr())
        }
    }

    fn pdf_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        if let Some(inner) = &self.inner {
            inner.pdf(wo, wi, swl, ctx)
        } else {
            0.0f32.expr()
        }
    }

    fn albedo_impl(
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

    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        if let Some(inner) = &self.inner {
            inner.roughness(wo, swl, ctx)
        } else {
            1.0f32.expr()
        }
    }

    fn emission_impl(
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
pub struct ScaledBsdf {
    pub inner: Rc<dyn Surface>,
    pub weight: Expr<f32>,
}
impl Surface for ScaledBsdf {
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.evaluate(wo, wi, swl, ctx) * self.weight
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

    fn pdf_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.inner.pdf(wo, wi, swl, ctx)
    }

    fn albedo_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.albedo(wo, swl, ctx) * self.weight
    }

    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.inner.roughness(wo, swl, ctx)
    }

    fn emission_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.emission(wo, swl, ctx) * self.weight
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
    #[tracked]
    fn evaluate_impl(
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
                let f_a = if frac.lt(1.0 - Self::EPS) {
                    self.bsdf_a.evaluate(wo, wi, swl, ctx)
                } else {
                    zero
                };
                let f_b = if frac.gt(Self::EPS) {
                    self.bsdf_b.evaluate(wo, wi, swl, ctx)
                } else {
                    zero
                };
                f_a * (1.0 - frac) + f_b * frac
            }
        }
    }
    #[tracked]
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        let (which, remapped) =
            weighted_discrete_choice2_and_remap(frac, 1u32.expr(), 0u32.expr(), u_select);
        if which.eq(0) {
            self.bsdf_a.sample_wi(wo, remapped, u_sample, swl, ctx)
        } else {
            self.bsdf_b.sample_wi(wo, remapped, u_sample, swl, ctx)
        }
    }
    #[tracked]
    fn pdf_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        let zero = 0.0f32.expr();
        let pdf_a = if frac.lt(1.0 - Self::EPS) {
            self.bsdf_a.pdf(wo, wi, swl, ctx)
        } else {
            zero
        };
        let pdf_b = if frac.gt(Self::EPS) {
            self.bsdf_b.pdf(wo, wi, swl, ctx)
        } else {
            zero
        };
        pdf_a * (1.0 - frac) + pdf_b * frac
    }
    #[tracked]
    fn albedo_impl(
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
    #[tracked]
    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        self.bsdf_a.roughness(wo, swl, ctx) * (1.0 - frac)
            + self.bsdf_b.roughness(wo, swl, ctx) * frac
    }
    #[tracked]
    fn emission_impl(
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
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        if debug_mode() {
            lc_assert!(wo.is_finite().all());
            lc_assert!(wi.is_finite().all());
        }
        self.inner
            .evaluate(self.frame.to_local(wo), self.frame.to_local(wi), swl, ctx)
    }
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let (wi, valid) =
            self.inner
                .sample_wi(self.frame.to_local(wo), u_select, u_sample, swl, ctx);
        (self.frame.to_world(wi), valid)
    }

    fn pdf_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.inner
            .pdf(self.frame.to_local(wo), self.frame.to_local(wi), swl, ctx)
    }
    fn albedo_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.albedo(self.frame.to_local(wo), swl, ctx)
    }
    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.inner.roughness(self.frame.to_local(wo), swl, ctx)
    }
    fn emission_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.emission(self.frame.to_local(wo), swl, ctx)
    }
}
impl SurfaceClosure {
    #[tracked]
    pub fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let wo = self.frame.to_local(wo);
        let (wi, valid) = self.inner.sample_wi(wo, u_select, u_sample, swl, ctx);
        let color = self.inner.evaluate(wo, wi, **swl, ctx);
        let pdf = self.inner.pdf(wo, wi, **swl, ctx);
        BsdfSample {
            wi: self.frame.to_world(wi),
            color,
            valid: valid & pdf.gt(0.0),
            pdf,
        }
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
    #[tracked]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let wh = wo + wi;
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        if (wh.dot(wo) * wi.dot(wh)).lt(0.0)
            | wh.eq(0.0).all()
            | cos_i.eq(0.0)
            | cos_o.eq(0.0)
            | !Frame::same_hemisphere(wo, wi)
        {
            Color::zero(ctx.color_repr)
        } else {
            let wh = wh.normalize();
            let f = self
                .fresnel
                .evaluate(wi.dot(face_forward(wh, Float3::expr(0.0, 1.0, 0.0))), ctx);
            let d = self.dist.d(wh, ctx.ad_mode);
            let g = self.dist.g(wo, wi, ctx.ad_mode);
            let f = &self.color * &f * (0.25 * d * g / (cos_i * cos_o)).abs() * cos_o.abs();
            f
        }
    }
    #[tracked]
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        _u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        _swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let wh = self.dist.sample_wh(wo, u_sample, ctx.ad_mode);
        let wi = reflect(wo, wh);
        let valid = Frame::same_hemisphere(wo, wi);
        (wi, valid)
    }
    #[tracked]
    fn pdf_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let wh = wo + wi;
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        if (wh.dot(wo) * wi.dot(wh)).lt(0.0)
            | wh.eq(0.0).all()
            | cos_i.eq(0.0)
            | cos_o.eq(0.0)
            | !Frame::same_hemisphere(wo, wi)
        {
            0.0f32.expr()
        } else {
            let wh = wh.normalize();
            // cpu_dbg!(wh);
            self.dist.pdf(wo, wh, ctx.ad_mode) / (4.0 * wo.dot(wh).abs())
        }
    }
    fn albedo_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.color
    }
    fn roughness_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.dist.roughness(ctx.ad_mode)
    }
    fn emission_impl(
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
    #[tracked]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(cos_o.gt(0.0), self.eta, 1.0 / self.eta);
        let wh = (wo + wi * eta).normalize();
        let wh = face_forward(wh, Float3::expr(0.0, 1.0, 0.0));
        let backfacing = (wh.dot(wi) * cos_i).lt(0.0) | (wh.dot(wo) * cos_o).lt(0.0);
        if (wh.dot(wo) * wi.dot(wh)).gt(0.0)
            | cos_i.eq(0.0)
            | cos_o.eq(0.0)
            | backfacing
            | Frame::same_hemisphere(wo, wi)
        {
            Color::zero(ctx.color_repr)
        } else {
            let f = self.fresnel.evaluate(wo.dot(wh), ctx);
            let denom = (wi.dot(wh) + wo.dot(wh) / eta).sqr() * cos_i * cos_o;
            select(
                denom.eq(0.0),
                Color::zero(ctx.color_repr),
                (Color::one(ctx.color_repr) - f)
                    * &self.color
                    * (self.dist.d(wh, ctx.ad_mode)
                        * self.dist.g(wo, wi, ctx.ad_mode)
                        * eta.sqr()
                        * wi.dot(wh).abs()
                        * wo.dot(wh).abs()
                        / denom)
                        .abs()
                    * cos_o.abs(),
            )

            // Expr<f32> denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap) * cosTheta_i * cosTheta_o;
            // Expr<f32> ft = mfDistrib.D(wm) * (1 - F) * mfDistrib.G(wo, wi) *
            //            std::abs(Dot(wi, wm) * Dot(wo, wm) / denom);
            // // Account for non-symmetry with transmission to different medium
            // if (mode == TransportMode::Radiance)
            //     ft /= Sqr(etap);

            // return SampledSpectrum(ft);
        }
    }
    #[tracked]
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        _u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let wh = self.dist.sample_wh(wo, u_sample, ctx.ad_mode);
        let (refracted, _eta, wi) = refract(wo, wh, self.eta);
        let valid = refracted & !Frame::same_hemisphere(wo, wi);
        if debug_mode() {
            let pdf = self.pdf(wo, wi, **swl, ctx);
            lc_assert!(pdf.ge(0.0));
        }
        (wi, valid)
    }
    #[tracked]
    fn pdf_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(cos_o.gt(0.0), self.eta, 1.0 / self.eta);
        let wh = (wo + wi * eta).normalize();
        let wh = face_forward(wh, Float3::expr(0.0, 1.0, 0.0));
        let backfacing = (wh.dot(wi) * cos_i).lt(0.0) | (wh.dot(wo) * cos_o).lt(0.0);
        if (wh.dot(wo) * wi.dot(wh)).gt(0.0)
            | cos_i.eq(0.0)
            | cos_o.eq(0.0)
            | backfacing
            | Frame::same_hemisphere(wo, wi)
        {
            0.0f32.expr()
        } else {
            // Expr<f32> denom = Sqr(Dot(wi, wm) + Dot(wo, wm) / etap);
            // Expr<f32> dwm_dwi = AbsDot(wi, wm) / denom;
            // pdf = mfDistrib.PDF(wo, wm) * dwm_dwi * pt / (pr + pt);
            let denom = (wi.dot(wh) + wo.dot(wh) / eta).sqr();
            let dwh_dwi = wi.dot(wh).abs() / denom;
            select(
                denom.eq(0.0),
                0.0f32.expr(),
                self.dist.pdf(wo, wh, ctx.ad_mode) * dwh_dwi,
            )
        }
    }
    fn albedo_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Color {
        self.color
    }
    fn roughness_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.dist.roughness(ctx.ad_mode)
    }
    fn emission_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        Color::zero(ctx.color_repr)
    }
}

#[tracked]
pub fn fr_dielectric(cos_theta_i: Expr<f32>, eta: Expr<f32>) -> Expr<f32> {
    let cos_theta_i = cos_theta_i.clamp(-1.0.expr(), 1.0.expr());
    let eta = select(cos_theta_i.gt(0.0), eta, 1.0 / eta);
    let cos_theta_i = cos_theta_i.abs();
    //
    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    let sin2_theta_i = 1.0 - cos_theta_i.sqr();
    let sin2_theta_t = sin2_theta_i / eta.sqr();
    if sin2_theta_t.ge(1.0) {
        1.0f32.expr()
    } else {
        let cos_theta_t = (1.0 - sin2_theta_t).max_(0.0).sqrt();
        let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
        let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
        let fr = (r_parl.sqr() + r_perp.sqr()) * 0.5;
        // cpu_dbg!(fr);
        fr.clamp(0.0.expr(), 1.0.expr())
    }
    //     Expr<f32> sin2Theta_i = 1 - Sqr(cosTheta_i);
    //     Expr<f32> sin2Theta_t = sin2Theta_i / Sqr(eta);
    //     if (sin2Theta_t >= 1)
    //         return 1.f;
    //     Expr<f32> cosTheta_t = SafeSqrt(1 - sin2Theta_t);

    //     Expr<f32> r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    //     Expr<f32> r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    //     return (Sqr(r_parl) + Sqr(r_perp)) / 2;
}
#[tracked]
pub fn fr_schlick(f0: Color, cos_theta_i: Expr<f32>) -> Color {
    let cos_theta_i = cos_theta_i.clamp(-1.0.expr(), 1.0.expr()).abs();
    let pow5 = |x: Expr<f32>| x.sqr().sqr() * x;
    let fr = f0 + (Color::one(f0.repr()) - f0) * pow5(1.0 - cos_theta_i);
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
    #[tracked]
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
