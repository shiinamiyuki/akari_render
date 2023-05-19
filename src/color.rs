use std::cell::Cell;

use crate::*;
pub mod colorspace {
    pub const SRGB: u32 = 0;
}
#[derive(Copy, Clone, Value)]
#[repr(C)]
pub struct SampledWavelengths {
    pub wavelengths: Float4,
    pub pdf: Float4,
}
#[derive(Copy, Clone)]
pub enum ColorRepr {
    Rgb,
    Spectral4,
}

#[derive(Aggregate, Clone, Copy)]
pub enum Color {
    Rgb(Expr<Float3>),
    Spectral4(Expr<Float4>),
}
#[derive(Aggregate, Clone, Copy)]
pub enum ColorVar {
    Rgb(Var<Float3>),
    Spectral4(Var<Float4>),
}

pub enum ColorBuffer {
    Rgb(Buffer<Float3>),
    Spectral4(Buffer<Float4>),
}
impl ColorBuffer {
    pub fn new(device: Device, count: usize, color_repr: ColorRepr) -> Self {
        match color_repr {
            ColorRepr::Spectral4 => ColorBuffer::Spectral4(device.create_buffer(count)),
            ColorRepr::Rgb => ColorBuffer::Rgb(device.create_buffer(count)),
        }
    }
    pub fn read(&self, i: impl Into<Expr<u32>>) -> Color {
        match self {
            ColorBuffer::Rgb(b) => Color::Rgb(b.var().read(i)),
            ColorBuffer::Spectral4(b) => Color::Spectral4(b.var().read(i)),
        }
    }
    pub fn write(&self, i: impl Into<Expr<u32>>, color: Color) {
        match self {
            ColorBuffer::Rgb(b) => b.var().write(i, color.as_rgb()),
            ColorBuffer::Spectral4(b) => b.var().write(i, color.as_spectral4()),
        }
    }
    pub fn as_rgb(&self) -> &Buffer<Float3> {
        match self {
            ColorBuffer::Rgb(b) => b,
            ColorBuffer::Spectral4(_) => panic!("as_rgb() called on spectral buffer"),
        }
    }
    pub fn as_spectral4(&self) -> &Buffer<Float4> {
        match self {
            ColorBuffer::Rgb(_) => panic!("as_spectral4() called on rgb buffer"),
            ColorBuffer::Spectral4(b) => b,
        }
    }
}
impl ColorVar {
    pub fn zero(repr: ColorRepr) -> Self {
        match repr {
            ColorRepr::Spectral4 => todo!(),
            ColorRepr::Rgb => ColorVar::Rgb(var!(Float3)),
        }
    }
    pub fn new(value: Color) -> Self {
        match value {
            Color::Rgb(v) => ColorVar::Rgb(var!(Float3, v)),
            Color::Spectral4(v) => ColorVar::Spectral4(var!(Float4, v)),
        }
    }
    pub fn one(repr: ColorRepr) -> Self {
        match repr {
            ColorRepr::Spectral4 => todo!(),
            ColorRepr::Rgb => ColorVar::Rgb(var!(Float3, Float3Expr::one())),
        }
    }
    pub fn repr(&self) -> ColorRepr {
        match self {
            ColorVar::Spectral4(s) => ColorRepr::Spectral4,
            ColorVar::Rgb(_) => ColorRepr::Rgb,
        }
    }
    pub fn load(&self) -> Color {
        match self {
            ColorVar::Spectral4(s) => todo!(),
            ColorVar::Rgb(v) => Color::Rgb(v.load()),
        }
    }
    pub fn store(&self, color: Color) {
        match self {
            ColorVar::Spectral4(s) => todo!(),
            ColorVar::Rgb(v) => v.store(color.as_rgb()),
        }
    }
}
impl Color {
    pub fn max(&self) -> Expr<f32> {
        match self {
            Color::Rgb(v) => v.reduce_max(),
            Color::Spectral4(_) => todo!(),
        }
    }
    pub fn into_dyn(&self) -> DynExpr {
        match self {
            Color::Rgb(rgb) => (*rgb).into(),
            Color::Spectral4(samples) => (*samples).into(),
        }
    }
    pub fn from_dyn(color: DynExpr, repr: ColorRepr) -> Self {
        match repr {
            ColorRepr::Rgb => {
                let rgb = color.get::<Float3>();
                Color::Rgb(rgb.xyz())
            }
            ColorRepr::Spectral4 => {
                let samples = color.get::<Float4>();
                Color::Spectral4(samples)
            }
        }
    }
    pub fn to_rgb(&self) -> Expr<Float3> {
        match self {
            Color::Rgb(rgb) => *rgb,
            Color::Spectral4(_) => {
                todo!()
            }
        }
    }
    pub fn as_rgb(&self) -> Expr<Float3> {
        match self {
            Color::Rgb(rgb) => *rgb,
            Color::Spectral4(_) => {
                panic!("not rgb!");
            }
        }
    }
    pub fn as_spectral4(&self) -> Expr<Float4> {
        match self {
            Color::Rgb(_) => {
                panic!("not spectral4!");
            }
            Color::Spectral4(samples) => *samples,
        }
    }
    pub fn zero(repr: ColorRepr) -> Color {
        match repr {
            ColorRepr::Spectral4 => todo!(),
            ColorRepr::Rgb => Color::Rgb(Float3Expr::zero()),
        }
    }
    pub fn one(repr: ColorRepr) -> Color {
        match repr {
            ColorRepr::Spectral4 => todo!(),
            ColorRepr::Rgb => Color::Rgb(Float3Expr::one()),
        }
    }
    pub fn repr(&self) -> ColorRepr {
        match self {
            Color::Spectral4(_) => ColorRepr::Spectral4,
            Color::Rgb(_) => ColorRepr::Rgb,
        }
    }
    pub fn max_component(&self) -> Float {
        match self {
            Color::Spectral4(s) => s.reduce_max(),
            Color::Rgb(s) => s.reduce_max(),
        }
    }
    pub fn has_nan(&self) -> Expr<bool> {
        match self {
            Color::Spectral4(s) => todo!(),
            Color::Rgb(v) => v.is_nan().any(),
        }
    }
    pub fn remove_nan(&self) -> Self {
        if_!(self.has_nan(), {
            Self::zero(self.repr())
        }, else {
            self.clone()
        })
    }
}
impl std::ops::Mul<Float> for &Color {
    type Output = Color;

    fn mul(self, rhs: Float) -> Self::Output {
        match self {
            Color::Spectral4(s) => Color::Spectral4(*s * rhs),
            Color::Rgb(s) => Color::Rgb(*s * rhs),
        }
    }
}
impl std::ops::Div<Float> for &Color {
    type Output = Color;

    fn div(self, rhs: Float) -> Self::Output {
        match self {
            Color::Spectral4(s) => Color::Spectral4(*s / rhs),
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
impl std::ops::Sub<&Color> for Color {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}
impl std::ops::Sub<Color> for Color {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}
impl std::ops::Mul<&Color> for &Color {
    type Output = Color;

    fn mul(self, rhs: &Color) -> Self::Output {
        match (self, rhs) {
            (Color::Spectral4(s), Color::Spectral4(t)) => Color::Spectral4(*s * *t),
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s * *t),
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}
impl std::ops::Add<&Color> for &Color {
    type Output = Color;

    fn add(self, rhs: &Color) -> Self::Output {
        match (self, rhs) {
            (Color::Spectral4(s), Color::Spectral4(t)) => Color::Spectral4(*s + *t),
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s + *t),
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}
impl std::ops::Sub<&Color> for &Color {
    type Output = Color;

    fn sub(self, rhs: &Color) -> Self::Output {
        match (self, rhs) {
            (Color::Spectral4(s), Color::Spectral4(t)) => Color::Spectral4(*s - *t),
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s - *t),
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
