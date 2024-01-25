use std::collections::HashMap;
use std::io::Write;
use std::rc::Rc;
use std::sync::Arc;

use akari_common::memmap2::{Mmap, MmapMut};
use scene_graph::NormalMapSpace;

use crate::color::{ColorRepr, ColorVar, SampledWavelengths};
use crate::geometry::{face_forward, reflect, refract, FrameComps, FrameExpr};
use crate::heap::MegaHeap;
use crate::microfacet::MicrofacetDistribution;
use crate::sampler::{init_pcg32_buffer_with_seed, IndependentSampler};
use crate::sampling::weighted_discrete_choice2_and_remap;
use crate::svm::eval::SvmEvaluator;
use crate::util::{polynomial, Complex};
use crate::{color::Color, geometry::Frame, *};

use super::Svm;

#[derive(Clone, Copy, Debug)]
pub struct BsdfEvalContext {
    pub color_repr: ColorRepr,
    pub ad_mode: ADMode,
}

pub mod diffuse;
pub mod glass;
pub mod metal;
pub mod plastic;
pub mod precompute;
pub mod principled;

#[derive(Clone, Aggregate)]
#[luisa(crate = "luisa")]
pub struct BsdfSample {
    pub wi: Expr<Float3>,
    pub pdf: Expr<f32>,
    pub color: Color,
    pub valid: Expr<bool>,
}
impl BsdfSample {
    pub fn invalid(color_repr: ColorRepr) -> Self {
        Self {
            wi: Expr::<Float3>::zeroed(),
            pdf: 0.0f32.expr(),
            color: Color::zero(color_repr),
            valid: false.expr(),
        }
    }
}

pub trait Surface {
    fn alpha(&self) -> Expr<f32> {
        1.0f32.expr()
    }

    fn ns(&self) -> Expr<Float3>;
    /// return f(wo, wi) * abs_cos_theta(wi), pdf
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>);
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>);
    fn albedo_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color;
    /// lobe roughness
    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
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
    ) -> (Color, Expr<f32>) {
        let ret_color = ColorVar::zero(ctx.color_repr);
        let ret_pdf = Var::<f32>::zeroed();
        maybe_outline(|| {
            let (color, pdf) = self.evaluate_impl(wo, wi, swl, ctx);
            ret_color.store(color);
            ret_pdf.store(pdf);
        });
        (ret_color.load(), ret_pdf.load())
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
        u_select: Expr<f32>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let ret = Var::<f32>::zeroed();
        maybe_outline(|| ret.store(self.roughness_impl(wo, u_select, swl, ctx)));
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
pub struct NullSurface {}
impl Surface for NullSurface {
    fn ns(&self) -> Expr<Float3> {
        Float3::expr(0.0, 0.0, 1.0)
    }
    fn evaluate_impl(
        &self,
        _wo: Expr<Float3>,
        _wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        (Color::zero(ctx.color_repr), 0.0f32.expr())
    }

    fn sample_wi_impl(
        &self,
        _wo: Expr<Float3>,
        _u_select: Expr<f32>,
        _u_sample: Expr<Float2>,
        _swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        (Expr::<Float3>::zeroed(), false.expr())
    }

    fn albedo_impl(
        &self,
        _wo: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        Color::zero(ctx.color_repr)
    }

    fn roughness_impl(
        &self,
        _wo: Expr<Float3>,
        _u_select: Expr<f32>,
        _swl: Expr<SampledWavelengths>,
        _ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        1.0f32.expr()
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
pub struct TransparentSurface {
    pub inner: Rc<dyn Surface>,
    pub alpha: Expr<f32>,
}
impl Surface for TransparentSurface {
    fn alpha(&self) -> Expr<f32> {
        self.alpha
    }
    fn ns(&self) -> Expr<Float3> {
        self.inner.ns()
    }
    #[tracked(crate = "luisa")]
    fn albedo_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.albedo(wo, swl, ctx) * self.alpha
    }
    fn emission_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        self.inner.emission(wo, swl, ctx) * self.alpha
    }
    #[tracked(crate = "luisa")]
    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let (which, remapped) =
            weighted_discrete_choice2_and_remap(self.alpha, 0u32.expr(), 1u32.expr(), u_select);
        if which == 0 {
            self.inner.roughness(wo, remapped, swl, ctx)
        } else {
            0.0f32.expr()
        }
    }
    #[tracked(crate = "luisa")]
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let (which, remapped) =
            weighted_discrete_choice2_and_remap(self.alpha, 0u32.expr(), 1u32.expr(), u_select);
        if which == 0 {
            self.inner.sample_wi(wo, remapped, u_sample, swl, ctx)
        } else {
            (-wo, true.expr())
        }
    }
    #[tracked(crate = "luisa")]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        let f_v = ColorVar::zero(ctx.color_repr);
        let pdf_v = 0.0f32.var();

        let almost_inf_pdf = 1e16f32.expr();
        let check_passthrough = |wo: Expr<Float3>, wi: Expr<Float3>| {
            // random test shows the max error is less than 6e-7, including local to world conversion
            (wo + wi).abs().reduce_max() < 6e-7
        };
        if self.alpha > 0.0 {
            let (f, pdf) = self.inner.evaluate(wo, wi, swl, ctx);
            if self.alpha < 1.0 {
                // random test shows the max error is less than 6e-7, including local to world conversion
                let is_passthrough = check_passthrough(wo, wi);
                if is_passthrough {
                    f_v.store(Color::one(ctx.color_repr) * almost_inf_pdf);
                    pdf_v.store(almost_inf_pdf);
                } else {
                    f_v.store(f);
                    pdf_v.store(pdf);
                }
            } else {
                f_v.store(f);
                pdf_v.store(pdf);
            }
        } else {
            let is_passthrough = check_passthrough(wo, wi);
            if is_passthrough {
                f_v.store(Color::one(ctx.color_repr) * almost_inf_pdf);
                pdf_v.store(almost_inf_pdf);
            }
        }
        (f_v.load(), pdf_v.load())
    }
}
pub struct EmissiveSurface {
    pub inner: Option<Rc<dyn Surface>>,
    pub emission: Color,
}
impl Surface for EmissiveSurface {
    fn alpha(&self) -> Expr<f32> {
        self.inner.as_ref().map_or(1.0f32.expr(), |s| s.alpha())
    }
    fn ns(&self) -> Expr<Float3> {
        self.inner
            .as_ref()
            .map_or(Float3::expr(0.0, 0.0, 1.0), |s| s.ns())
    }
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        if let Some(inner) = &self.inner {
            inner.evaluate(wo, wi, swl, ctx)
        } else {
            (Color::zero(ctx.color_repr), 0.0f32.expr())
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
        u_select: Expr<f32>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        if let Some(inner) = &self.inner {
            inner.roughness(wo, u_select, swl, ctx)
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
    pub weight: Box<dyn Fn(Expr<Float3>, &BsdfEvalContext) -> Color>,
}
impl Surface for ScaledBsdf {
    fn alpha(&self) -> Expr<f32> {
        self.inner.alpha()
    }
    fn ns(&self) -> Expr<Float3> {
        self.inner.ns()
    }
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        let (color, pdf) = self.inner.evaluate(wo, wi, swl, ctx);
        let weight = (self.weight)(wo, ctx);
        (color * weight, pdf)
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

    fn albedo_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let weight = (self.weight)(wo, ctx);
        self.inner.albedo(wo, swl, ctx) * weight
    }

    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.inner.roughness(wo, u_select, swl, ctx)
    }

    fn emission_impl(
        &self,
        wo: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Color {
        let weight = (self.weight)(wo, ctx);
        self.inner.emission(wo, swl, ctx) * weight
    }
}
pub struct BsdfMixture {
    /// `frac` controls how two bsdfs are mixed:
    /// - Under [`BsdfBlendMode::Addictive`], frac is used as a MIS weight
    /// - Under [`BsdfBlendMode::Mix`], frac is used to linearly interpolate between two bsdfs
    ///
    /// *Note:* if frac depends on wo and mode is [`BsdfBlendMode::Mix`], then the Bsdf is not symmetric
    pub frac: Box<dyn Fn(Expr<Float3>, &BsdfEvalContext) -> Expr<f32>>,
    pub bsdf_a: Rc<dyn Surface>,
    pub bsdf_b: Rc<dyn Surface>,
    pub mode: BsdfBlendMode,
}
impl BsdfMixture {
    const EPS: f32 = 1e-4;
}

impl Surface for BsdfMixture {
    fn alpha(&self) -> Expr<f32> {
        self.bsdf_a.alpha().max_(self.bsdf_b.alpha())
    }
    #[tracked(crate = "luisa")]
    fn ns(&self) -> Expr<Float3> {
        (self.bsdf_a.ns() + self.bsdf_b.ns()).normalize()
    }
    #[tracked(crate = "luisa")]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        let frac = (self.frac)(wo, ctx);
        match self.mode {
            BsdfBlendMode::Addictive => {
                let (f_a, pdf_a) = self.bsdf_a.evaluate(wo, wi, swl, ctx);
                let (f_b, pdf_b) = self.bsdf_b.evaluate(wo, wi, swl, ctx);
                (f_a + f_b, pdf_a.lerp(pdf_b, frac))
            }
            BsdfBlendMode::Mix => {
                let zero = (Color::zero(ctx.color_repr), 0.0f32.expr());
                let (f_a, pdf_a) = if frac.lt(1.0 - Self::EPS) {
                    self.bsdf_a.evaluate(wo, wi, swl, ctx)
                } else {
                    zero
                };
                let (f_b, pdf_b) = if frac.gt(Self::EPS) {
                    self.bsdf_b.evaluate(wo, wi, swl, ctx)
                } else {
                    zero
                };
                (f_a.lerp_f32(f_b, frac), pdf_a.lerp(pdf_b, frac))
            }
        }
    }

    #[tracked(crate = "luisa")]
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
    #[tracked(crate = "luisa")]
    fn roughness_impl(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        let frac: Expr<f32> = (self.frac)(wo, ctx);
        let (which, remapped) =
            weighted_discrete_choice2_and_remap(frac, 1u32.expr(), 0u32.expr(), u_select);
        if which.eq(0) {
            self.bsdf_a.roughness(wo, remapped, swl, ctx)
        } else {
            self.bsdf_b.roughness(wo, remapped, swl, ctx)
        }
    }
    #[tracked(crate = "luisa")]
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

    #[tracked(crate = "luisa")]
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
    pub ng: Expr<Float3>,
}
impl SurfaceClosure {
    // prevents light leaking
    // copied from LuisaRender
    #[tracked(crate = "luisa")]
    fn check_wo_wi_valid(&self, wo: Expr<Float3>, wi: Expr<Float3>) -> Expr<bool> {
        let sign = |x: Expr<f32>| (x > 0.0).select(1.0f32.expr(), -1.0f32.expr());
        let ns = self.frame.n;
        let ng = self.ng;
        let flipped = sign(ng.dot(ns));

        (sign(flipped * wo.dot(ns)) * sign(wo.dot(ng)) > 0.0)
            & (sign(flipped * wi.dot(ns)) * sign(wi.dot(ng)) > 0.0)
        // let reflected_s = wo.dot(ns) * wi.dot(ns) > 0.0;
        // let reflected_g = wo.dot(ng) * wi.dot(ng) > 0.0;
        // reflected_s == reflected_g
        // false.expr()
    }
}
impl Surface for SurfaceClosure {
    fn alpha(&self) -> Expr<f32> {
        self.inner.alpha()
    }
    fn ns(&self) -> Expr<Float3> {
        let ns = self.inner.ns();
        self.frame.to_world(ns)
    }
    // return f(wo, wi) * abs_cos_theta(wi)
    #[tracked(crate = "luisa")]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        if debug_mode() {
            lc_assert!(wo.is_finite().all());
            lc_assert!(wi.is_finite().all());
        }
        let valid = self.check_wo_wi_valid(wo, wi);
        if !valid {
            (Color::zero(ctx.color_repr), 0.0f32.expr())
        } else {
            self.inner
                .evaluate(self.frame.to_local(wo), self.frame.to_local(wi), swl, ctx)
        }
    }
    #[tracked(crate = "luisa")]
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
        let wi = self.frame.to_world(wi);
        let valid = valid & self.check_wo_wi_valid(wo, wi);
        (wi, valid)
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
        u_select: Expr<f32>,
        swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> Expr<f32> {
        self.inner
            .roughness(self.frame.to_local(wo), u_select, swl, ctx)
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
    #[tracked(crate = "luisa")]
    pub fn sample(
        &self,
        wo: Expr<Float3>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> BsdfSample {
        let (wi, valid) = self.sample_wi(wo, u_select, u_sample, swl, ctx);
        if !valid {
            BsdfSample::invalid(ctx.color_repr)
        } else {
            let (color, pdf) = self.evaluate(wo, wi, **swl, ctx);
            BsdfSample {
                wi,
                color,
                valid: valid & pdf.gt(0.0),
                pdf,
            }
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
    fn ns(&self) -> Expr<Float3> {
        Float3::expr(0.0, 0.0, 1.0)
    }
    #[tracked(crate = "luisa")]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        let wh = wo + wi;
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        if (wh.dot(wo) * wi.dot(wh)).lt(0.0)
            | wh.eq(0.0).all()
            | cos_i.eq(0.0)
            | cos_o.eq(0.0)
            | !Frame::same_hemisphere(wo, wi)
        {
            (Color::zero(ctx.color_repr), 0.0f32.expr())
        } else {
            let wh = wh.normalize();
            let f = self
                .fresnel
                .evaluate(wi.dot(face_forward(wh, Float3::expr(0.0, 0.0, 1.0))), ctx);
            let d = self.dist.d(wh, ctx.ad_mode);
            let g = self.dist.g(wo, wi, ctx.ad_mode);
            let f = &self.color * &f * (0.25 * d * g / (cos_i * cos_o)).abs() * cos_i.abs();
            let pdf = self.dist.pdf(wo, wh, ctx.ad_mode) / (4.0 * wo.dot(wh).abs());
            (f, pdf)
        }
    }
    #[tracked(crate = "luisa")]
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
        _u_select: Expr<f32>,
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
    fn ns(&self) -> Expr<Float3> {
        Float3::expr(0.0, 0.0, 1.0)
    }
    #[tracked(crate = "luisa")]
    fn evaluate_impl(
        &self,
        wo: Expr<Float3>,
        wi: Expr<Float3>,
        _swl: Expr<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Color, Expr<f32>) {
        let cos_o = Frame::cos_theta(wo);
        let cos_i = Frame::cos_theta(wi);
        let eta = select(cos_o.gt(0.0), self.eta, 1.0 / self.eta);
        let wh = (wo + wi * eta).normalize();
        let wh = face_forward(wh, Float3::expr(0.0, 0.0, 1.0));
        let backfacing = (wh.dot(wi) * cos_i).lt(0.0) | (wh.dot(wo) * cos_o).lt(0.0);
        if (wh.dot(wo) * wi.dot(wh)).gt(0.0)
            | cos_i.eq(0.0)
            | cos_o.eq(0.0)
            | backfacing
            | Frame::same_hemisphere(wo, wi)
        {
            (Color::zero(ctx.color_repr), 0.0f32.expr())
        } else {
            let f = {
                let f = self.fresnel.evaluate(wo.dot(wh), ctx);
                let denom = (wi.dot(wh) + wo.dot(wh) / eta).sqr() * cos_i * cos_o;
                // device_log!("eta: {}", eta);
                select(
                    denom.eq(0.0),
                    Color::zero(ctx.color_repr),
                    (Color::one(ctx.color_repr) - f)
                        * &self.color
                        * (self.dist.d(wh, ctx.ad_mode)
                            * self.dist.g(wo, wi, ctx.ad_mode)
                            / eta.sqr()
                            * wi.dot(wh).abs()
                            * wo.dot(wh).abs()
                            / denom)
                            .abs()
                        * cos_i.abs(),
                )
            };
            let pdf = {
                let denom = (wi.dot(wh) + wo.dot(wh) / eta).sqr();
                let dwh_dwi = wi.dot(wh).abs() / denom;
                select(
                    denom.eq(0.0),
                    0.0f32.expr(),
                    self.dist.pdf(wo, wh, ctx.ad_mode) * dwh_dwi,
                )
            };
            (f, pdf)
        }
    }

    #[tracked(crate = "luisa")]
    fn sample_wi_impl(
        &self,
        wo: Expr<Float3>,
        _u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        _swl: Var<SampledWavelengths>,
        ctx: &BsdfEvalContext,
    ) -> (Expr<Float3>, Expr<bool>) {
        let wh = self.dist.sample_wh(wo, u_sample, ctx.ad_mode);
        let (refracted, _eta, wi) = refract(wo, wh, self.eta);
        let valid = refracted & !Frame::same_hemisphere(wo, wi);
        (wi, valid)
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
        _u_select: Expr<f32>,
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

#[tracked(crate = "luisa")]
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

/// Gulbrandsen's parametrization
#[tracked(crate = "luisa")]
pub fn artistic_to_conductor_fresnel(color: Color, tint: Color) -> (Color, Color) {
    let r = color.clamp(0.99f32.expr());
    let g = tint;
    let r_sqrt = r.sqrt();
    let one = Color::one(color.repr());
    let n_min = (one - r) / (one + r);
    let n_max = (one + r_sqrt) / (one - r_sqrt);
    let n = n_max.lerp(n_min, g);
    let k2 = ((n + one) * (n + one) * r - (n - one) * (n - one)) / (one - r);
    let k2 = k2.map(|x| x.max_(0.0f32.expr()));
    let k = k2.sqrt();
    (n, k)
}

#[tracked(crate = "luisa")]
pub fn fr_complex(cos_theta_i: Expr<f32>, eta: Complex) -> Expr<f32> {
    let cos_theta_i = cos_theta_i.clamp(0.0.expr(), 0.999.expr());
    let sin2_theta = 1.0 - cos_theta_i.sqr();
    let sin2_theta_t = Complex::new(sin2_theta, 0.0) / eta.sqr();
    let cos_theta_t = (Complex::new(1.0, 0.0) - sin2_theta_t).sqrt();

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (Complex::new(cos_theta_i, 0.0) - eta * cos_theta_t)
        / (Complex::new(cos_theta_i, 0.0) + eta * cos_theta_t);
    let f = (r_parl.norm() + r_perp.norm()) * 0.5;
    // lc_assert!(f.is_finite());
    f
}

#[tracked(crate = "luisa")]
pub fn fr_complex_spec(cos_theta_i: Expr<f32>, n: Color, k: Color) -> Color {
    match (n, k) {
        (Color::Rgb(n, cs0), Color::Rgb(k, cs1)) => {
            assert_eq!(cs0, cs1);
            let f_r = fr_complex(cos_theta_i, Complex::new(n.r, k.r));
            let f_g = fr_complex(cos_theta_i, Complex::new(n.g, k.g));
            let f_b = fr_complex(cos_theta_i, Complex::new(n.b, k.b));
            Color::Rgb(Float3::expr(f_r, f_g, f_b), cs0)
        }
        _ => todo!(),
    }
}
// #[tracked(crate = "luisa")]
// pub fn ior_from_specular(specular:Expr<f32>)->Expr<f32> {
//     let s = specular * 0.08;
//     let s = s.clamp(0.0.expr(), 0.99.expr()).sqrt();
//     (1.0 + s) / (1.0 - s)
// }
/// Taken from Cycles
#[tracked(crate = "luisa")]
pub fn ior_from_f0(f0: Expr<f32>) -> Expr<f32> {
    let sqrt_f0 = f0.clamp(0.0.expr(), 0.99.expr()).sqrt();
    (1.0 + sqrt_f0) / (1.0 - sqrt_f0)
}
#[tracked(crate = "luisa")]
pub fn f0_from_ior(ior: Expr<f32>) -> Expr<f32> {
    let f0 = (ior - 1.0) / (ior + 1.0);
    f0.sqr()
}
/// Taken from Cycles
#[tracked(crate = "luisa")]
pub fn ior_parametrization(z: Expr<f32>) -> Expr<f32> {
    ior_from_f0(z.sqr().sqr())
}
#[tracked(crate = "luisa")]
pub fn fr_schlick(f0: Color, f90: Color, cos_theta_i: Expr<f32>) -> Color {
    let cos_theta_i = cos_theta_i.clamp(-1.0.expr(), 1.0.expr()).abs();
    let pow5 = |x: Expr<f32>| x.sqr().sqr() * x;
    let fr = f0 + (f90 - f0) * pow5(1.0 - cos_theta_i);
    fr
}
#[tracked(crate = "luisa")]
pub fn fr_generalized_schlick(
    f0: Color,
    f90: Color,
    cos_theta_i: Expr<f32>,
    exponent: Expr<f32>,
) -> Color {
    if exponent == 5.0 {
        fr_schlick(f0, f90, cos_theta_i)
    } else {
        let cos_theta_i = cos_theta_i.clamp(-1.0.expr(), 1.0.expr()).abs();
        let fr = f0 + (f90 - f0) * (1.0 - cos_theta_i).powf(exponent);
        fr
    }
}
#[tracked(crate = "luisa")]
pub fn fr_dielectric_integral(eta: Expr<f32>) -> Expr<f32> {
    if eta == 1.0 {
        0.0f32.expr()
    } else if eta < 1.0 {
        let coeffs: [f32; 4] = [0.75985009f32, -2.09069066f32, 2.23559031f32, -0.90663979f32];
        polynomial(
            eta,
            &std::array::from_fn::<Expr<f32>, 4, _>(|i| coeffs[i].expr()),
        )
    } else {
        let coeffs: [f32; 3] = [0.97945724f32, 0.21762732f32, -1.18995376f32];
        polynomial(
            1.0 / eta,
            &std::array::from_fn::<Expr<f32>, 3, _>(|i| coeffs[i].expr()),
        )
    }
}
#[tracked(crate = "luisa")]
pub fn ggx_dielectric_albedo(
    table: &PreComputedTable,
    roughness: Expr<f32>,
    cos_theta_i: Expr<f32>,
    eta: Expr<f32>,
) -> Expr<f32> {
    let z = ((eta - 1.0) / (eta + 1.0)).abs().sqrt();
    let cos_theta_i = cos_theta_i.clamp(-0.999.expr(), 0.999.expr()).abs();
    table.read_3d(roughness, cos_theta_i.abs(), z)
}
#[derive(Copy, Clone)]
pub struct FresnelGeneralizedSchlick {
    pub f0: Color,
    pub f90: Color,
    pub exponent: Expr<f32>,
}
impl Fresnel for FresnelGeneralizedSchlick {
    fn evaluate(&self, cos_theta_i: Expr<f32>, _ctx: &BsdfEvalContext) -> Color {
        fr_generalized_schlick(self.f0, self.f90, cos_theta_i, self.exponent)
    }
}
#[derive(Copy, Clone)]
pub struct FresnelDielectric {
    pub eta: Expr<f32>, //eta = eta_t / eta_i
}
impl Fresnel for FresnelDielectric {
    #[tracked(crate = "luisa")]
    fn evaluate(&self, cos_theta_i: Expr<f32>, ctx: &BsdfEvalContext) -> Color {
        Color::one(ctx.color_repr) * fr_dielectric(cos_theta_i, self.eta)
    }
}

#[derive(Copy, Clone)]
pub struct FresnelComplex {
    pub n: Color,
    pub k: Color,
}
impl Fresnel for FresnelComplex {
    #[tracked(crate = "luisa")]
    fn evaluate(&self, cos_theta_i: Expr<f32>, _: &BsdfEvalContext) -> Color {
        fr_complex_spec(cos_theta_i.abs(), self.n, self.k)
    }
}
#[derive(Copy, Clone)]
pub struct ConstFresnel {}
impl Fresnel for ConstFresnel {
    fn evaluate(&self, _: Expr<f32>, ctx: &BsdfEvalContext) -> Color {
        Color::one(ctx.color_repr)
    }
}
#[derive(Clone)]
pub struct PreComputedTable {
    heap: Arc<MegaHeap>,
    pub dim: [u32; 3],
    pub idx: u32,
}
pub struct PreComputeOptions<F> {
    pub samples: u32,
    pub dim: [u32; 3],
    pub f: F,
}
pub struct PreComputedTables {
    device: Device,
    heap: Arc<MegaHeap>,
    name_to_heap_idx: HashMap<String, PreComputedTable>,
}
impl PreComputedTable {
    #[tracked(crate = "luisa")]
    fn _read_1d(
        buf: &BindlessBufferVar<f32>,
        x: Expr<f32>,
        offset: Expr<u32>,
        size: Expr<u32>,
    ) -> Expr<f32> {
        let x = x.clamp(0.0f32.expr(), 1.0f32.expr()) * (size.as_f32() - 1.0);
        let index = x.floor().as_u32();
        let nindex = (index + 1).min_(size - 1);
        let t = x - index.as_f32();
        let data0 = buf.read(offset + index);
        let data1 = buf.read(offset + nindex);
        (1.0 - t) * data0 + t * data1
    }
    #[tracked(crate = "luisa")]
    fn _read_2d(
        buf: &BindlessBufferVar<f32>,
        x: Expr<f32>,
        y: Expr<f32>,
        offset: Expr<u32>,
        xsize: Expr<u32>,
        ysize: Expr<u32>,
    ) -> Expr<f32> {
        let y = y.clamp(0.0f32.expr(), 1.0f32.expr()) * (ysize.as_f32() - 1.0);
        let index = y.floor().as_u32();
        let nindex = (index + 1).min_(ysize - 1);
        let t = y - index.as_f32();
        let data0 = Self::_read_1d(buf, x, offset + xsize * index, xsize);
        let data1 = Self::_read_1d(buf, x, offset + xsize * nindex, xsize);
        (1.0 - t) * data0 + t * data1
    }
    #[tracked(crate = "luisa")]
    fn _read_3d(
        buf: &BindlessBufferVar<f32>,
        x: Expr<f32>,
        y: Expr<f32>,
        z: Expr<f32>,
        offset: Expr<u32>,
        xsize: Expr<u32>,
        ysize: Expr<u32>,
        zsize: Expr<u32>,
    ) -> Expr<f32> {
        let z = z.clamp(0.0f32.expr(), 1.0f32.expr()) * (zsize.as_f32() - 1.0);
        let index = z.floor().as_u32();
        let nindex = (index + 1).min_(zsize - 1);
        let t = z - index.as_f32();
        let data0 = Self::_read_2d(buf, x, y, offset + xsize * ysize * index, xsize, ysize);
        let data1 = Self::_read_2d(buf, x, y, offset + xsize * ysize * nindex, xsize, ysize);
        (1.0 - t) * data0 + t * data1
    }
    #[tracked(crate = "luisa")]
    pub fn read_3d(&self, x: Expr<f32>, y: Expr<f32>, z: Expr<f32>) -> Expr<f32> {
        let buf = self.heap.buffer(self.idx);
        Self::_read_3d(
            &buf,
            x,
            y,
            z,
            0u32.expr(),
            self.dim[0].expr(),
            self.dim[1].expr(),
            self.dim[2].expr(),
        )
    }
    #[tracked(crate = "luisa")]
    pub fn read_2d(&self, x: Expr<f32>, y: Expr<f32>) -> Expr<f32> {
        let buf = self.heap.buffer(self.idx);
        Self::_read_2d(
            &buf,
            x,
            y,
            0u32.expr(),
            self.dim[0].expr(),
            self.dim[1].expr(),
        )
    }
    #[tracked(crate = "luisa")]
    pub fn read_1d(&self, x: Expr<f32>) -> Expr<f32> {
        let buf = self.heap.buffer(self.idx);
        Self::_read_1d(&buf, x, 0u32.expr(), self.dim[0].expr())
    }
}
impl PreComputedTables {
    pub fn new(device: Device, heap: Arc<MegaHeap>) -> Self {
        Self {
            heap,
            name_to_heap_idx: HashMap::new(),
            device,
        }
    }
    pub fn get(&self, name: impl AsRef<str>) -> Option<PreComputedTable> {
        self.name_to_heap_idx.get(name.as_ref()).cloned()
    }
    pub fn get_or_compute<F>(
        &mut self,
        name: impl AsRef<str>,
        opt: PreComputeOptions<F>,
    ) -> PreComputedTable
    where
        F: Fn(Expr<Float3>, &IndependentSampler) -> Expr<f32>,
    {
        let exe_path = std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
        let exe_dir = exe_path.parent().unwrap();
        let cache_file = exe_dir.join(format!("{}.precomputed", name.as_ref()));
        let n_floats = opt.dim[0] as usize * opt.dim[1] as usize * opt.dim[2] as usize;
        if let Ok(file) = std::fs::File::open(&cache_file) {
            log::info!("Loading precomputed table from {}", cache_file.display());
            let bytes = unsafe { Mmap::map(&file).unwrap() };

            assert_eq!(bytes.len(), n_floats * std::mem::size_of::<f32>());
            let slice =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n_floats) };
            let buf = self.device.create_buffer_from_slice(slice);
            let table = PreComputedTable {
                heap: self.heap.clone(),
                dim: opt.dim,
                idx: self.heap.bind_buffer(&buf),
            };
            self.name_to_heap_idx
                .insert(name.as_ref().to_string(), table.clone());
            table
        } else {
            log::info!("Precomputing table {}", name.as_ref());
            let pcg = init_pcg32_buffer_with_seed(self.device.clone(), n_floats, 0);
            let table = self.device.create_buffer::<f32>(n_floats);
            let precompute_kernel = self.device.create_kernel::<fn()>(&track!(|| {
                set_block_size([4, 4, 4]);
                let tid = dispatch_id();
                let global_id = tid.x + tid.y * opt.dim[0] + tid.z * (opt.dim[0] * opt.dim[1]);
                let rng = IndependentSampler::from_pcg32(pcg.read(global_id).var());
                let fx = (tid.x.as_f32() / (opt.dim[0] as f32 - 1.0))
                    .clamp(1e-4f32.expr(), 0.9999f32.expr());
                let fy = (tid.y.as_f32() / (opt.dim[1] as f32 - 1.0))
                    .clamp(1e-4f32.expr(), 0.9999f32.expr());
                let fz = (tid.z.as_f32() / (opt.dim[2] as f32 - 1.0))
                    .clamp(1e-4f32.expr(), 0.9999f32.expr());
                let sum = 0.0f32.var();
                for _ in 0..opt.samples {
                    *sum += (opt.f)(Float3::expr(fx, fy, fz), &rng);
                }
                let avg = sum / opt.samples as f32;
                table.write(global_id, avg);
            }));
            precompute_kernel.dispatch([opt.dim[0], opt.dim[1], opt.dim[2]]);
            {
                let mut file = std::fs::File::create(&cache_file).unwrap();
                let data = table.copy_to_vec();
                file.write_all(unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<f32>(),
                    )
                })
                .unwrap();
            }
            let table = PreComputedTable {
                heap: self.heap.clone(),
                dim: opt.dim,
                idx: self.heap.bind_buffer(&table),
            };
            self.name_to_heap_idx
                .insert(name.as_ref().to_string(), table.clone());
            table
        }
    }
}
#[tracked(crate = "luisa")]
pub fn normal_map(
    surface: Rc<dyn Surface>,
    ns: Expr<Float3>,
    ng: Expr<Float3>,
    original_frame: Expr<Frame>,
    space: NormalMapSpace,
) -> Rc<dyn Surface> {
    let normal = ns;
    let frame = original_frame;
    let new_frame = match space {
        NormalMapSpace::TangentSpace => {
            if (normal == 0.0).all() {
                FrameExpr::identity()
            } else {
                let tt = frame.t;
                let normal = normal.normalize();
                let n_world = frame.to_world(normal);
                // todo: clamp shading normal
                let new_frame = FrameExpr::from_n_t(n_world, tt);
                let new_frame_t_local = frame.to_local(new_frame.t);
                let new_frame_s_local = frame.to_local(new_frame.s);
                let new_frame_n_local = frame.to_local(new_frame.n);
                Frame::from_comps_expr(FrameComps {
                    n: new_frame_n_local,
                    s: new_frame_s_local,
                    t: new_frame_t_local,
                })
            }
        }
        NormalMapSpace::ObjectSpace => todo!(),
        NormalMapSpace::WorldSpace => todo!(),
    };
    Rc::new(SurfaceClosure {
        inner: surface,
        frame: new_frame,
        ng: original_frame.to_local(ng),
    }) as Rc<dyn Surface>
}
