use crate::{*, color::FlatColor, surface::Bsdf};
use nodes::*;

pub enum Value {
    Float(Expr<f32>),
    Float3(Expr<Float3>),
    Spectrum(Expr<FlatColor>),
    Closure(Box<dyn Bsdf>),
}

pub struct Compiler {
    
}