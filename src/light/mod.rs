use crate::{*, util::alias_table::AliasTable};
pub trait Light {}


pub trait LightDistribution {
    fn sample(&self, u: Expr<Float2>) -> (Uint, Float);
    fn pdf(&self, light_index: Uint) -> Float;
}

pub struct WeightedDistribution {
    pub alias_table: AliasTable,
}

impl LightDistribution for WeightedDistribution {
    fn sample(&self, u: Expr<Float2>) -> (Uint, Float) {
        self.alias_table.sample(u)
    }
    fn pdf(&self, light_index: Uint) -> Float {
        self.alias_table.pdf(light_index)
    }
}