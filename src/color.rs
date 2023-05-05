use std::cell::Cell;

use crate::*;
pub mod colorspace {
    pub const SRGB: u32 = 0;
}
#[derive(Aggregate, Clone)]
pub struct SampledWavelengths {
    pub lambda: Vec<Float>,
    pub pdf: Vec<Cell<Float>>,
}

impl SampledWavelengths {
    pub fn new(lambda: Vec<Float>, pdf: Vec<Cell<Float>>) -> Self {
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
    pub samples: Vec<Float>,
    pub wavelengths: SampledWavelengths,
}
#[derive(Aggregate, Clone)]
pub enum ColorRepr {
    Rgb,
    Spectral(SampledWavelengths),
}

#[derive(Aggregate, Clone)]
pub enum Color {
    Rgb(Expr<Float3>),
    Spectral(SampledSpectrum),
}
impl Color {
    pub fn zero(repr: &ColorRepr) -> Color {
        match repr {
            ColorRepr::Spectral(s) => Color::Spectral(SampledSpectrum {
                wavelengths: s.clone(),
                samples: vec![Float::from(0.0); s.nsamples()],
            }),
            ColorRepr::Rgb => Color::Rgb(Float3Expr::zero()),
        }
    }
    pub fn one(repr: &ColorRepr) -> Color {
        match repr {
            ColorRepr::Spectral(s) => Color::Spectral(SampledSpectrum {
                wavelengths: s.clone(),
                samples: vec![Float::from(1.0); s.nsamples()],
            }),
            ColorRepr::Rgb => Color::Rgb(Float3Expr::one()),
        }
    }
    pub fn repr(&self) -> ColorRepr {
        match self {
            Color::Spectral(s) => ColorRepr::Spectral(s.wavelengths.clone()),
            Color::Rgb(_) => ColorRepr::Rgb,
        }
    }
    pub fn max_component(&self) -> Float {
        match self {
            Color::Spectral(s) => s
                .samples
                .iter()
                .fold(Float::from(0.0), |acc, x| acc.max(*x)),
            Color::Rgb(s) => s.reduce_max(),
        }
    }
}
impl std::ops::Mul<Float> for &Color {
    type Output = Color;

    fn mul(self, rhs: Float) -> Self::Output {
        match self {
            Color::Spectral(s) => Color::Spectral(SampledSpectrum {
                samples: s.samples.iter().map(|x| *x * rhs).collect(),
                wavelengths: s.wavelengths.clone(),
            }),
            Color::Rgb(s) => Color::Rgb(*s * rhs),
        }
    }
}
impl std::ops::Div<Float> for &Color {
    type Output = Color;

    fn div(self, rhs: Float) -> Self::Output {
        match self {
            Color::Spectral(s) => Color::Spectral(SampledSpectrum {
                samples: s.samples.iter().map(|x| *x / rhs).collect(),
                wavelengths: s.wavelengths.clone(),
            }),
            Color::Rgb(s) => Color::Rgb(*s / rhs),
        }
    }
}
impl std::ops::Mul<Float> for Color {
    type Output = Self;
    fn mul(self, rhs: Float) -> Self::Output {
        &self * rhs
    }
}
impl std::ops::Div<Float> for Color {
    type Output = Self;
    fn div(self, rhs: Float) -> Self::Output {
        &self / rhs
    }
}
impl std::ops::Mul<&Color> for Color {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}
impl std::ops::Mul<Color> for Color {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}
impl std::ops::Add<&Color> for Color {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}
impl std::ops::Add<Color> for Color {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}
impl std::ops::Mul<&Color> for &Color {
    type Output = Color;

    fn mul(self, rhs: &Color) -> Self::Output {
        match (self, rhs) {
            (Color::Spectral(s), Color::Spectral(t)) => {
                Color::Spectral(SampledSpectrum {
                    samples: s
                        .samples
                        .iter()
                        .zip(t.samples.iter())
                        .map(|(x, y)| *x * *y)
                        .collect(),
                    wavelengths: s.wavelengths.clone(),
                })
            }
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s * *t),
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}
impl std::ops::Add<&Color> for &Color {
    type Output = Color;

    fn add(self, rhs: &Color) -> Self::Output {
        match (self, rhs) {
            (Color::Spectral(s), Color::Spectral(t)) => {
                Color::Spectral(SampledSpectrum {
                    samples: s
                        .samples
                        .iter()
                        .zip(t.samples.iter())
                        .map(|(x, y)| *x + *y)
                        .collect(),
                    wavelengths: s.wavelengths.clone(),
                })
            }
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s + *t),
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}



pub fn glam_srgb_to_linear(rgb: glam::Vec3) -> glam::Vec3 {
    fn f32_srgb_to_linear1(s: f32) -> f32 {
        if s <= 0.04045 {
            s / 12.92
        } else {
            (((s + 0.055) / 1.055) as f32).powf(2.4)
        }
    }
    glam::vec3(
        f32_srgb_to_linear1(rgb.x),
        f32_srgb_to_linear1(rgb.y),
        f32_srgb_to_linear1(rgb.z),
    )
}