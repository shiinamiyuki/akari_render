use crate::*;
pub trait Sampler {
    fn next_1d(&self) -> Float;
    fn next_2d(&self) -> Expr<Float2>;
}
