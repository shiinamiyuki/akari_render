use crate::{*, util::alias_table::AliasTable};
pub trait Light {}


pub trait LightDistribution {
    fn sample(&self, u: Expr<Vec2>) -> (Uint32, Float32);
    fn pdf(&self, light_index: Uint32) -> Float32;
}

pub struct WeightedDistribution {
    pub alias_table: AliasTable,
}

impl LightDistribution for WeightedDistribution {
    fn sample(&self, u: Expr<Vec2>) -> (Uint32, Float32) {
        self.alias_table.sample(u)
    }
    fn pdf(&self, light_index: Uint32) -> Float32 {
        self.alias_table.pdf(light_index)
    }
}