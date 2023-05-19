use std::sync::Arc;

use crate::{
    color::{Color, ColorRepr},
    geometry::{PointNormal, Ray},
    interaction::SurfaceInteraction,
    mesh::MeshAggregate,
    sampler::Sampler,
    texture::TextureEvaluator,
    util::alias_table::AliasTable,
    *,
};

#[derive(Clone, Aggregate)]
pub struct LightSample {
    pub li: Color,
    pub pdf: Float,
    pub wi: Expr<Float3>,
    pub shadow_ray: Expr<Ray>,
    pub n: Expr<Float3>,
}
#[derive(Clone, Copy, Value)]
#[repr(C)]
pub struct TLightSample<T: Value> {
    pub li: T,
    pub pdf: f32,
    pub wi: Float3,
    pub shadow_ray: Ray,
    pub n: Float3,
}
pub struct LightEvalContext<'a> {
    pub texture: &'a TextureEvaluator,
    pub color_repr: ColorRepr,
    pub meshes: &'a MeshAggregate,
}

pub trait Light {
    fn id(&self) -> Expr<u32>;
    fn le(&self, ray: Expr<Ray>, si: Expr<SurfaceInteraction>, ctx: &LightEvalContext<'_>)
        -> Color;
    fn sample_direct(
        &self,
        pn: Expr<PointNormal>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        ctx: &LightEvalContext<'_>,
    ) -> LightSample;
    fn pdf_direct(
        &self,
        si: Expr<SurfaceInteraction>,
        pn: Expr<PointNormal>,
        ctx: &LightEvalContext<'_>,
    ) -> Expr<f32>;
}

pub trait LightDistribution {
    fn sample_and_remap(&self, u: Expr<f32>) -> (Uint, Expr<f32>, Expr<f32>);
    fn pdf(&self, light_index: Uint) -> Float;
}

pub struct WeightedLightDistribution {
    pub alias_table: AliasTable,
}

impl LightDistribution for WeightedLightDistribution {
    fn sample_and_remap(&self, u: Expr<f32>) -> (Uint, Expr<f32>, Expr<f32>) {
        self.alias_table.sample_and_remap(u)
    }
    fn pdf(&self, light_index: Uint) -> Float {
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

// represents all lights in the scene as a *single* light
pub struct LightAggregate {
    pub lights: Polymorphic<PolyKey, dyn Light>,
    pub light_ids_to_lights: Buffer<TagIndex>,
    pub light_distribution: Box<dyn LightDistribution>,
    pub meshes: Arc<MeshAggregate>,
}
pub struct LightEvaluator {
    pub(crate) color_repr: ColorRepr,
    pub(crate) le: DynCallable<(Expr<Ray>, Expr<SurfaceInteraction>), DynExpr>,
    pub(crate) pdf: Callable<(Expr<SurfaceInteraction>, Expr<PointNormal>), Expr<f32>>,
    pub(crate) sample: DynCallable<(Expr<PointNormal>, Expr<Float3>), DynExpr>,
}
impl LightEvaluator {
    pub fn le(&self, ray: Expr<Ray>, si: Expr<SurfaceInteraction>) -> Color {
        let color = self.le.call(ray, si);
        Color::from_dyn(color, self.color_repr)
    }
    pub fn sample(&self, pn: Expr<PointNormal>, u: Expr<Float3>) -> LightSample {
        let sample = self.sample.call(pn, u);
        match self.color_repr {
            ColorRepr::Rgb => {
                let sample = sample.get::<TLightSample<Float3>>();
                LightSample {
                    li: Color::Rgb(sample.li()),
                    pdf: sample.pdf(),
                    wi: sample.wi(),
                    shadow_ray: sample.shadow_ray(),
                    n: sample.n(),
                }
            }
            _ => todo!(),
        }
    }
    pub fn pdf(&self, si: Expr<SurfaceInteraction>, pn: Expr<PointNormal>) -> Expr<f32> {
        self.pdf.call(si, pn)
    }
}
impl LightAggregate {
    pub fn light(&self, si: Expr<SurfaceInteraction>) -> PolymorphicRef<PolyKey, dyn Light> {
        let inst_id = si.inst_id();
        let instance = self.meshes.mesh_instances.var().read(inst_id);
        let light = self.lights.get(instance.light());
        light
    }
    pub fn le(
        &self,
        ray: Expr<Ray>,
        si: Expr<SurfaceInteraction>,
        ctx: &LightEvalContext<'_>,
    ) -> Color {
        let light = self.light(si);
        let light_choice_pdf = self
            .light_distribution
            .pdf(light.dispatch(|_tag, _key, light| light.id()));
        let direct = light.dispatch(|_tag, _key, light| light.le(ray, si, ctx));
        direct
    }
    pub fn sample_direct(
        &self,
        pn: Expr<PointNormal>,
        u_select: Expr<f32>,
        u_sample: Expr<Float2>,
        ctx: &LightEvalContext<'_>,
    ) -> LightSample {
        let light_dist = &self.light_distribution;
        let (light_idx, light_choice_pdf, u_select) = light_dist.sample_and_remap(u_select);
        let light = self.light_ids_to_lights.var().read(light_idx);
        let light = self.lights.get(light);
        let mut sample =
            light.dispatch(|_, _, light| light.sample_direct(pn, u_select, u_sample, ctx));
        sample.pdf = sample.pdf * light_choice_pdf;
        sample
    }
    pub fn pdf_direct(
        &self,
        si: Expr<SurfaceInteraction>,
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
