use std::sync::Arc;

use crate::{
    color::{Color, ColorPipeline, ColorRepr, FlatColor, SampledWavelengths},
    geometry::{PointNormal, Ray},
    interaction::SurfaceInteraction,
    mesh::MeshAggregate,
    svm::{surface::BsdfEvalContext, Svm},
    util::distribution::AliasTable,
    *,
};

#[derive(Clone, Aggregate)]
pub struct LightSample {
    pub li: Color,
    pub pdf: Expr<f32>,
    pub wi: Expr<Float3>,
    pub shadow_ray: Expr<Ray>,
    pub n: Expr<Float3>,
    pub valid: Expr<bool>,
}
pub struct LightEvalContext<'a> {
    pub color_pipeline: ColorPipeline,
    pub meshes: &'a MeshAggregate,
    pub svm: &'a Svm,
    pub surface_eval_ctx: &'a BsdfEvalContext<'a>,
}
impl<'a> LightEvalContext<'a> {
    pub fn color_repr(&self) -> ColorRepr {
        self.color_pipeline.color_repr
    }
}

pub trait Light {
    fn id(&self) -> Expr<u32>;
    fn le(
        &self,
        ray: Expr<Ray>,
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> Color;
    fn sample_direct(
        &self,
        pn: Expr<PointNormal>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> LightSample;
    fn pdf_direct(
        &self,
        si: SurfaceInteraction,
        pn: Expr<PointNormal>,
        ctx: &LightEvalContext<'_>,
    ) -> Expr<f32>;
}

pub trait LightDistribution: Send + Sync {
    fn sample_and_remap(&self, u: Expr<f32>) -> (Expr<u32>, Expr<f32>, Expr<f32>);
    fn pdf(&self, light_index: Expr<u32>) -> Expr<f32>;
}

pub struct WeightedLightDistribution {
    pub alias_table: AliasTable,
}

impl LightDistribution for WeightedLightDistribution {
    fn sample_and_remap(&self, u: Expr<f32>) -> (Expr<u32>, Expr<f32>, Expr<f32>) {
        self.alias_table.sample_and_remap(u)
    }
    fn pdf(&self, light_index: Expr<u32>) -> Expr<f32> {
        self.alias_table.pdf(light_index)
    }
}
impl WeightedLightDistribution {
    pub fn new(device: Device, weights: &[f32]) -> Self {
        let alias_table = AliasTable::new(device, weights);
        Self { alias_table }
    }
}
pub mod area;
pub mod point;

// represents all lights in the scene as a *single* light
pub struct LightAggregate {
    pub lights: Polymorphic<(), dyn Light>,
    pub light_ids_to_lights: Buffer<TagIndex>,
    pub light_distribution: Box<dyn LightDistribution>,
    pub meshes: Arc<MeshAggregate>,
}
impl LightAggregate {
    pub fn light(&self, si: SurfaceInteraction) -> PolymorphicRef<(), dyn Light> {
        let inst_id = si.inst_id;
        let instance = self.meshes.mesh_instances().read(inst_id);
        let light = self.lights.get(instance.light);
        light
    }
    pub fn le(
        &self,
        ray: Expr<Ray>,
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> Color {
        let light = self.light(si);
        let _light_choice_pdf = self
            .light_distribution
            .pdf(light.dispatch(|_tag, _key, light| light.id()));
        let direct = light.dispatch(|_tag, _key, light| light.le(ray, si, swl, ctx));
        direct
    }
    #[tracked]
    pub fn sample_direct(
        &self,
        pn: Expr<PointNormal>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        swl: Expr<SampledWavelengths>,
        ctx: &LightEvalContext<'_>,
    ) -> LightSample {
        let light_dist = &self.light_distribution;
        let (light_idx, light_choice_pdf, u_select) = light_dist.sample_and_remap(u_select);
        // let (light_idx, light_choice_pdf) = (0u32.expr(), 1.0f32.expr());
        let light = self.light_ids_to_lights.var().read(light_idx);
        let light = self.lights.get(light);
        let mut sample =
            light.dispatch(|_, _, light| light.sample_direct(pn, u_select, u_sample, swl, ctx));
        sample.pdf = sample.pdf * light_choice_pdf;
        sample
    }
    #[tracked]
    pub fn pdf_direct(
        &self,
        si: SurfaceInteraction,
        pn: Expr<PointNormal>,
        ctx: &LightEvalContext<'_>,
    ) -> Expr<f32> {
        let light = self.light(si);
        let light_choice_pdf = self
            .light_distribution
            .pdf(light.dispatch(|_tag, _key, light| light.id()));
        let light_pdf =
            light_choice_pdf * light.dispatch(|_tag, _key, light| light.pdf_direct(si, pn, ctx));
        light_pdf
    }
}
