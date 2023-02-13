use std::cell::Cell;

use crate::*;
pub mod colorspace {
    pub const SRGB: u32 = 0;
}
#[derive(Aggregate, Clone)]
pub struct SampledWavelengths {
    pub lambda: Vec<Float32>,
    pub pdf: Vec<Cell<Float32>>,
}

impl SampledWavelengths {
    pub fn new(lambda: Vec<Float32>, pdf: Vec<Cell<Float32>>) -> Self {
        Self { lambda, pdf }
    }
    pub fn is_empty(&self) -> bool {
        self.lambda.is_empty()
    }
    pub fn nsamples(&self) -> usize {
        self.lambda.len()
    }
}
#[derive(Aggregate, Clone)]
pub struct SampledSpectrum {
    pub samples:Vec<Float32>,
    pub wavelengths: SampledWavelengths,
}
#[derive(Aggregate, Clone)]
pub enum ColorRepr {
    Rgb,
    Spectral(SampledWavelengths)
}

#[derive(Aggregate, Clone)]
pub enum Color {
    Rgb(Expr<Vec3>),
    Spectral(SampledSpectrum)
}