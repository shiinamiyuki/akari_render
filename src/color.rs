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
pub const N_WAVELENGTH_SAMPLES: usize = 4;
#[derive(Aggregate, Clone)]
pub struct SampledSpectrum {
    pub samples: Vec<Float>,
    pub wavelengths: SampledWavelengths,
}
#[derive(Aggregate, Clone)]
pub struct SampledSpectrumVar {
    pub samples: Vec<Var<f32>>,
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
#[derive(Aggregate, Clone)]
pub enum ColorVar {
    Rgb(Var<Float3>),
    Spectral(SampledSpectrumVar),
}
impl ColorVar {
    pub fn zero(repr: &ColorRepr) -> Self {
        match repr {
            ColorRepr::Spectral(s) => ColorVar::Spectral(SampledSpectrumVar {
                wavelengths: s.clone(),
                samples: (0..s.nsamples()).map(|_| var!(f32)).collect(),
            }),
            ColorRepr::Rgb => ColorVar::Rgb(var!(Float3)),
        }
    }
    pub fn one(repr: &ColorRepr) -> Self {
        match repr {
            ColorRepr::Spectral(s) => ColorVar::Spectral(SampledSpectrumVar {
                wavelengths: s.clone(),
                samples: (0..s.nsamples()).map(|_| var!(f32, 1.0)).collect(),
            }),
            ColorRepr::Rgb => ColorVar::Rgb(var!(Float3, Float3Expr::one())),
        }
    }
    pub fn repr(&self) -> ColorRepr {
        match self {
            ColorVar::Spectral(s) => ColorRepr::Spectral(s.wavelengths.clone()),
            ColorVar::Rgb(_) => ColorRepr::Rgb,
        }
    }
    pub fn load(&self) -> Color {
        match self {
            ColorVar::Spectral(s) => Color::Spectral(SampledSpectrum {
                samples: todo!(),
                wavelengths: todo!(),
            }),
            ColorVar::Rgb(v) => Color::Rgb(v.load()),
        }
    }
    pub fn store(&self, color: &Color) {
        match self {
            ColorVar::Spectral(s) => todo!(),
            ColorVar::Rgb(v) => v.store(color.as_rgb()),
        }
    }
}
impl Color {
    pub fn to_rgb(&self) -> Expr<Float3> {
        match self {
            Color::Rgb(rgb) => *rgb,
            Color::Spectral(_) => {
                todo!()
            }
        }
    }
    pub fn as_rgb(&self) -> Expr<Float3> {
        match self {
            Color::Rgb(rgb) => *rgb,
            Color::Spectral(_) => {
                panic!("not rgb!");
            }
        }
    }
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
            (Color::Spectral(s), Color::Spectral(t)) => Color::Spectral(SampledSpectrum {
                samples: s
                    .samples
                    .iter()
                    .zip(t.samples.iter())
                    .map(|(x, y)| *x * *y)
                    .collect(),
                wavelengths: s.wavelengths.clone(),
            }),
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s * *t),
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}
impl std::ops::Add<&Color> for &Color {
    type Output = Color;

    fn add(self, rhs: &Color) -> Self::Output {
        match (self, rhs) {
            (Color::Spectral(s), Color::Spectral(t)) => Color::Spectral(SampledSpectrum {
                samples: s
                    .samples
                    .iter()
                    .zip(t.samples.iter())
                    .map(|(x, y)| *x + *y)
                    .collect(),
                wavelengths: s.wavelengths.clone(),
            }),
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s + *t),
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}

#[inline]
pub fn f32_srgb_to_linear1(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        (((s + 0.055) / 1.055) as f32).powf(2.4)
    }
}
#[inline]
pub fn glam_srgb_to_linear(rgb: glam::Vec3) -> glam::Vec3 {
    glam::vec3(
        f32_srgb_to_linear1(rgb.x),
        f32_srgb_to_linear1(rgb.y),
        f32_srgb_to_linear1(rgb.z),
    )
}
#[inline]
pub fn f32_linear_to_srgb1(l: f32) -> f32 {
    if l <= 0.0031308 {
        l * 12.92
    } else {
        l.powf(1.0 / 2.4) * 1.055 - 0.055
    }
}
#[inline]
pub fn glam_linear_to_srgb(linear: glam::Vec3) -> glam::Vec3 {
    glam::vec3(
        f32_linear_to_srgb1(linear.x),
        f32_linear_to_srgb1(linear.y),
        f32_linear_to_srgb1(linear.z),
    )
}
pub const XYZ_TO_SRGB: glam::Mat3 = glam::mat3(
    glam::vec3(3.240479f32, -0.969256f32, 0.055648f32),
    glam::vec3(-1.537150f32, 1.875991f32, -0.204043f32),
    glam::vec3(-0.498535f32, 0.041556f32, 1.057311f32),
);
pub const SRGB_TO_XYZ: glam::Mat3 = glam::mat3(
    glam::vec3(0.412453, 0.212671, 0.019334),
    glam::vec3(0.357580, 0.715160, 0.119193),
    glam::vec3(0.180423, 0.072169, 0.950227),
);
