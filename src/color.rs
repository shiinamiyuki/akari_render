use crate::*;
pub mod colorspace {
    pub const SRGB: u32 = 0;
}
#[derive(Copy, Clone, Value, Debug)]
#[repr(C)]
pub struct SampledWavelengths {
    pub wavelengths: Float4,
    pub pdf: Float4,
}
#[derive(Copy, Clone)]
pub enum ColorRepr {
    Rgb,
    Spectral,
}

#[derive(Aggregate, Clone, Copy)]
pub enum Color {
    Rgb(Expr<Float3>),
    Spectral(Expr<FlatColor>),
}
#[derive(Aggregate, Clone, Copy)]
pub enum ColorVar {
    Rgb(Var<Float3>),
    Spectral(Var<FlatColor>),
}

// flattened color values for storage
#[derive(Copy, Clone, Value, Debug)]
#[repr(C)]
pub struct FlatColor {
    pub samples: Float4,
    pub wavelengths: SampledWavelengths,
}

pub enum ColorBuffer {
    Rgb(Buffer<Float3>),
    Spectral(Buffer<FlatColor>),
}
impl ColorBuffer {
    pub fn new(device: Device, count: usize, color_repr: ColorRepr) -> Self {
        match color_repr {
            ColorRepr::Spectral => ColorBuffer::Spectral(device.create_buffer(count)),
            ColorRepr::Rgb => ColorBuffer::Rgb(device.create_buffer(count)),
        }
    }
    pub fn read(&self, i: impl Into<Expr<u32>>) -> Color {
        match self {
            ColorBuffer::Rgb(b) => Color::Rgb(b.var().read(i)),
            ColorBuffer::Spectral(b) => Color::Spectral(b.var().read(i)),
        }
    }
    pub fn write(&self, i: impl Into<Expr<u32>>, color: Color) {
        match self {
            ColorBuffer::Rgb(b) => b.var().write(i, color.as_rgb()),
            ColorBuffer::Spectral(b) => b.var().write(i, color.as_spectral()),
        }
    }
    pub fn as_rgb(&self) -> &Buffer<Float3> {
        match self {
            ColorBuffer::Rgb(b) => b,
            ColorBuffer::Spectral(_) => panic!("as_rgb() called on spectral buffer"),
        }
    }
    pub fn as_spectral(&self) -> &Buffer<FlatColor> {
        match self {
            ColorBuffer::Rgb(_) => panic!("as_Spectral() called on rgb buffer"),
            ColorBuffer::Spectral(b) => b,
        }
    }
}
impl ColorVar {
    pub fn zero(repr: ColorRepr) -> Self {
        match repr {
            ColorRepr::Spectral => todo!(),
            ColorRepr::Rgb => ColorVar::Rgb(var!(Float3)),
        }
    }
    pub fn new(value: Color) -> Self {
        match value {
            Color::Rgb(v) => ColorVar::Rgb(var!(Float3, v)),
            Color::Spectral(v) => ColorVar::Spectral(var!(FlatColor, v)),
        }
    }
    pub fn one(repr: ColorRepr) -> Self {
        match repr {
            ColorRepr::Spectral => todo!(),
            ColorRepr::Rgb => ColorVar::Rgb(var!(Float3, Float3Expr::one())),
        }
    }
    pub fn repr(&self) -> ColorRepr {
        match self {
            ColorVar::Spectral(_s) => ColorRepr::Spectral,
            ColorVar::Rgb(_) => ColorRepr::Rgb,
        }
    }
    pub fn load(&self) -> Color {
        match self {
            ColorVar::Spectral(_s) => todo!(),
            ColorVar::Rgb(v) => Color::Rgb(v.load()),
        }
    }
    pub fn store(&self, color: Color) {
        match self {
            ColorVar::Spectral(_s) => todo!(),
            ColorVar::Rgb(v) => v.store(color.as_rgb()),
        }
    }
}
impl Color {
    pub fn max(&self) -> Expr<f32> {
        match self {
            Color::Rgb(v) => v.reduce_max(),
            Color::Spectral(_) => todo!(),
        }
    }
    pub fn min(&self) -> Expr<f32> {
        match self {
            Color::Rgb(v) => v.reduce_min(),
            Color::Spectral(_) => todo!(),
        }
    }
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
    pub fn as_spectral(&self) -> Expr<FlatColor> {
        match self {
            Color::Rgb(_) => {
                panic!("not spectral!");
            }
            Color::Spectral(samples) => *samples,
        }
    }
    pub fn flatten(&self) -> Expr<FlatColor> {
        match self {
            Color::Rgb(rgb) => FlatColorExpr::new(
                make_float4(rgb.x(), rgb.y(), rgb.z(), 0.0),
                zeroed::<SampledWavelengths>(),
            ),
            Color::Spectral(samples) => *samples,
        }
    }
    pub fn from_flat(repr: ColorRepr, color: Expr<FlatColor>) -> Self {
        match repr {
            ColorRepr::Rgb => Color::Rgb(color.samples().xyz()),
            ColorRepr::Spectral => Color::Spectral(color),
        }
    }
    pub fn zero(repr: ColorRepr) -> Color {
        match repr {
            ColorRepr::Spectral => todo!(),
            ColorRepr::Rgb => Color::Rgb(Float3Expr::zero()),
        }
    }
    pub fn one(repr: ColorRepr) -> Color {
        match repr {
            ColorRepr::Spectral => todo!(),
            ColorRepr::Rgb => Color::Rgb(Float3Expr::one()),
        }
    }
    pub fn repr(&self) -> ColorRepr {
        match self {
            Color::Spectral(_) => ColorRepr::Spectral,
            Color::Rgb(_) => ColorRepr::Rgb,
        }
    }
    pub fn has_nan(&self) -> Expr<bool> {
        match self {
            Color::Spectral(_s) => todo!(),
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
    pub fn clamp(&self, max_contrib: impl Into<Expr<f32>>) -> Self {
        let max_contrib = max_contrib.into();
        match self {
            Color::Spectral(_s) => todo!(),
            Color::Rgb(v) => {
                Color::Rgb(v.clamp(Float3Expr::zero(), Float3Expr::splat(max_contrib)))
            }
        }
    }
}
impl std::ops::Mul<Float> for &Color {
    type Output = Color;

    fn mul(self, rhs: Float) -> Self::Output {
        match self {
            Color::Spectral(s) => Color::Spectral(s.set_samples(s.samples() * rhs)),
            Color::Rgb(s) => Color::Rgb(*s * rhs),
        }
    }
}
impl std::ops::Div<Float> for &Color {
    type Output = Color;

    fn div(self, rhs: Float) -> Self::Output {
        match self {
            Color::Spectral(s) => Color::Spectral(s.set_samples(s.samples() / rhs)),
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
        &self - rhs
    }
}
impl std::ops::Sub<Color> for Color {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}
impl std::ops::Mul<&Color> for &Color {
    type Output = Color;

    fn mul(self, rhs: &Color) -> Self::Output {
        match (self, rhs) {
            (Color::Spectral(s), Color::Spectral(t)) => {
                Color::Spectral(s.set_samples(s.samples() * t.samples()))
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
                Color::Spectral(s.set_samples(s.samples() + t.samples()))
            }
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s + *t),
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}
impl std::ops::Sub<&Color> for &Color {
    type Output = Color;

    fn sub(self, rhs: &Color) -> Self::Output {
        match (self, rhs) {
            (Color::Spectral(s), Color::Spectral(t)) => {
                Color::Spectral(s.set_samples(s.samples() - t.samples()))
            }
            (Color::Rgb(s), Color::Rgb(t)) => Color::Rgb(*s - *t),
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}
pub fn srgb_to_linear(rgb: Float3Expr) -> Float3Expr {
    Float3Expr::select(
        rgb.cmple(0.04045),
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).powf(2.4),
    )
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
