use crate::{
    color::Color,
    geometry::{PointNormal, Ray},
    interaction::{ShadingContext, SurfaceInteraction},
    sampler::Sampler,
    util::alias_table::AliasTable,
    *,
};

#[derive(Clone, Aggregate)]
pub struct LightSample {
    pub li: Color,
    pub pdf: Float,
    pub shadow_ray: Expr<Ray>,
    pub n: Expr<Float3>,
}
pub trait Light {
    fn le(&self, ray: Expr<Ray>, si: Expr<SurfaceInteraction>, ctx: &ShadingContext<'_>) -> Color;
    fn sample_direct(
        &self,
        pn: Expr<PointNormal>,
        u: Expr<Float4>,
        ctx: &ShadingContext<'_>,
    ) -> LightSample;
}

pub trait LightDistribution {
    fn sample(&self, u: Expr<Float2>) -> (Uint, Float);
    fn pdf(&self, light_index: Uint) -> Float;
}

pub struct WeightedLightDistribution {
    pub alias_table: AliasTable,
}

impl LightDistribution for WeightedLightDistribution {
    fn sample(&self, u: Expr<Float2>) -> (Uint, Float) {
        self.alias_table.sample(u)
    }
    fn pdf(&self, light_index: Uint) -> Float {
        self.alias_table.pdf(light_index)
    }
}
impl WeightedLightDistribution {
    pub fn new(device: Device, weights: &[f32]) -> luisa::Result<Self> {
        let alias_table = AliasTable::new(device, weights)?;
        Ok(Self { alias_table })
    }
}
pub mod area;
