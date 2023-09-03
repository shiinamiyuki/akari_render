use crate::*;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Aggregate, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub enum RgbColorSpace {
    #[serde(rename = "srgb")]
    SRgb,
    #[serde(rename = "aces")]
    ACEScg,
}
impl From<nodes::ColorSpace> for RgbColorSpace {
    fn from(value: nodes::ColorSpace) -> Self {
        match value {
            nodes::ColorSpace::ACEScg => RgbColorSpace::ACEScg,
            nodes::ColorSpace::SRGB => RgbColorSpace::SRgb,
        }
    }
}
pub struct ColorSpaceId;
impl ColorSpaceId {
    pub const NONE: u32 = 0;
    pub const SRGB: u32 = 1;
    pub const ACES_CG: u32 = 2;
    pub fn from_colorspace(colorspace: RgbColorSpace) -> u32 {
        match colorspace {
            RgbColorSpace::ACEScg => ColorSpaceId::ACES_CG,
            RgbColorSpace::SRgb => ColorSpaceId::SRGB,
        }
    }
    pub fn to_colorspace(id: u32) -> RgbColorSpace {
        match id {
            ColorSpaceId::ACES_CG => RgbColorSpace::ACEScg,
            ColorSpaceId::SRGB => RgbColorSpace::SRgb,
            _ => panic!("invalid colorspace id"),
        }
    }
}

#[derive(Copy, Clone, Value, Debug)]
#[repr(C)]
pub struct SampledWavelengths {
    pub wavelengths: Float4,
    pub pdf: Float4,
}
impl SampledWavelengthsExpr {
    pub fn rgb_wavelengths() -> Self {
        struct_!(SampledWavelengths {
            wavelengths: make_float4(0.0, 0.0, 0.0, 0.0),
            pdf: make_float4(1.0, 1.0, 1.0, 1.0),
        })
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ColorRepr {
    #[serde(rename = "rgb")]
    Rgb(RgbColorSpace),
    #[serde(rename = "spectral")]
    Spectral,
}
impl ColorRepr {
    pub fn rgb_colorspace(&self) -> Option<RgbColorSpace> {
        match self {
            ColorRepr::Rgb(cs) => Some(*cs),
            _ => None,
        }
    }
}

#[derive(Aggregate, Clone, Copy)]
pub enum Color {
    Rgb(Expr<Float3>, RgbColorSpace),
    Spectral(Expr<Float4>),
}
#[derive(Aggregate, Clone, Copy)]
pub enum ColorVar {
    Rgb(Var<Float3>, RgbColorSpace),
    Spectral(Var<Float4>),
}

// flattened color values for storage
pub type FlatColor = Float4;
pub type FlatColorExpr = Expr<Float4>;

#[derive(Copy, Clone, Value, Debug)]
#[repr(C)]
pub struct SampledSpectrum {
    pub samples: Float4,
    pub wavelengths: SampledWavelengths,
}

pub enum ColorBuffer {
    Rgb(Buffer<Float3>, RgbColorSpace),
    Spectral(Buffer<SampledSpectrum>),
}
impl ColorBuffer {
    pub fn new(device: Device, count: usize, color_repr: ColorRepr) -> Self {
        match color_repr {
            ColorRepr::Spectral => ColorBuffer::Spectral(device.create_buffer(count)),
            ColorRepr::Rgb(colorspace) => ColorBuffer::Rgb(device.create_buffer(count), colorspace),
        }
    }
    pub fn read(&self, i: impl IntoIndex) -> (Color, Expr<SampledWavelengths>) {
        match self {
            ColorBuffer::Rgb(b, cs) => (
                Color::Rgb(b.var().read(i), *cs),
                SampledWavelengthsExpr::rgb_wavelengths(),
            ),
            ColorBuffer::Spectral(b) => {
                let c = b.var().read(i);
                (Color::Spectral(c.samples()), c.wavelengths())
            }
        }
    }
    pub fn write(&self, i: impl IntoIndex, color: Color, swl: Expr<SampledWavelengths>) {
        match self {
            ColorBuffer::Rgb(b, cs) => b.var().write(i, color.as_rgb()),
            ColorBuffer::Spectral(b) => b.var().write(i, color.as_sampled_spectrum(swl)),
        }
    }
    pub fn as_rgb(&self) -> &Buffer<Float3> {
        match self {
            ColorBuffer::Rgb(b, _) => b,
            ColorBuffer::Spectral(_) => panic!("as_rgb() called on spectral buffer"),
        }
    }
    pub fn as_spectral(&self) -> &Buffer<SampledSpectrum> {
        match self {
            ColorBuffer::Rgb(_, _) => panic!("as_Spectral() called on rgb buffer"),
            ColorBuffer::Spectral(b) => b,
        }
    }
}
impl ColorVar {
    pub fn zero(repr: ColorRepr) -> Self {
        match repr {
            ColorRepr::Spectral => ColorVar::Spectral(var!(Float4)),
            ColorRepr::Rgb(cs) => ColorVar::Rgb(var!(Float3), cs),
        }
    }
    pub fn new(value: Color) -> Self {
        match value {
            Color::Rgb(v, cs) => ColorVar::Rgb(var!(Float3, v), cs),
            Color::Spectral(v) => ColorVar::Spectral(var!(Float4, v)),
        }
    }
    pub fn one(repr: ColorRepr) -> Self {
        match repr {
            ColorRepr::Spectral => ColorVar::Spectral(var!(Float4, Float4Expr::one())),
            ColorRepr::Rgb(cs) => ColorVar::Rgb(var!(Float3, Float3Expr::one()), cs),
        }
    }
    pub fn repr(&self) -> ColorRepr {
        match self {
            ColorVar::Spectral(_s) => ColorRepr::Spectral,
            ColorVar::Rgb(_, cs) => ColorRepr::Rgb(*cs),
        }
    }
    pub fn load(&self) -> Color {
        match self {
            ColorVar::Spectral(s) => Color::Spectral(**s),
            ColorVar::Rgb(v, cs) => Color::Rgb(v.load(), *cs),
        }
    }
    pub fn store(&self, color: Color) {
        match self {
            ColorVar::Spectral(s) => s.store(color.as_spectral()),
            ColorVar::Rgb(v, cs) => v.store(color.as_rgb()),
        }
    }
}
impl Color {
    pub fn max(&self) -> Expr<f32> {
        match self {
            Color::Rgb(v, _) => v.reduce_max(),
            Color::Spectral(_) => todo!(),
        }
    }
    pub fn min(&self) -> Expr<f32> {
        match self {
            Color::Rgb(v, _) => v.reduce_min(),
            Color::Spectral(_) => todo!(),
        }
    }
    pub fn to_rgb(&self, colorspace: RgbColorSpace) -> Expr<Float3> {
        match self {
            Color::Rgb(rgb, cs) => {
                if *cs == colorspace {
                    *rgb
                } else {
                    match (cs, colorspace) {
                        (RgbColorSpace::SRgb, RgbColorSpace::ACEScg) => {
                            const_(Mat3::from(srgb_to_aces_with_cat_mat())) * *rgb
                        }
                        (RgbColorSpace::ACEScg, RgbColorSpace::SRgb) => {
                            const_(Mat3::from(aces_to_srgb_with_cat_mat())) * *rgb
                        }
                        _ => unreachable!(),
                    }
                }
            }
            Color::Spectral(_) => {
                todo!()
            }
        }
    }
    pub fn as_rgb(&self) -> Expr<Float3> {
        match self {
            Color::Rgb(rgb, _) => *rgb,
            Color::Spectral(_) => {
                panic!("not rgb!");
            }
        }
    }
    pub fn as_sampled_spectrum(&self, swl: Expr<SampledWavelengths>) -> Expr<SampledSpectrum> {
        match self {
            Color::Rgb(_, _) => {
                panic!("not spectral!");
            }
            Color::Spectral(samples) => SampledSpectrumExpr::new(*samples, swl),
        }
    }
    pub fn as_spectral(&self) -> Expr<Float4> {
        match self {
            Color::Rgb(_, _) => {
                panic!("not spectral!");
            }
            Color::Spectral(samples) => *samples,
        }
    }
    pub fn flatten(&self) -> Expr<FlatColor> {
        match self {
            Color::Rgb(rgb, _) => make_float4(rgb.x(), rgb.y(), rgb.z(), 0.0),
            Color::Spectral(samples) => *samples,
        }
    }
    pub fn from_flat(repr: ColorRepr, color: Expr<FlatColor>) -> Self {
        match repr {
            ColorRepr::Rgb(cs) => Color::Rgb(color.xyz(), cs),
            ColorRepr::Spectral => Color::Spectral(color),
        }
    }
    pub fn zero(repr: ColorRepr) -> Color {
        match repr {
            ColorRepr::Spectral => todo!(),
            ColorRepr::Rgb(cs) => Color::Rgb(Float3Expr::zero(), cs),
        }
    }
    pub fn one(repr: ColorRepr) -> Color {
        match repr {
            ColorRepr::Spectral => todo!(),
            ColorRepr::Rgb(cs) => Color::Rgb(Float3Expr::one(), cs),
        }
    }
    pub fn repr(&self) -> ColorRepr {
        match self {
            Color::Spectral(_) => ColorRepr::Spectral,
            Color::Rgb(_, cs) => ColorRepr::Rgb(*cs),
        }
    }
    pub fn has_nan(&self) -> Expr<bool> {
        match self {
            Color::Spectral(_s) => todo!(),
            Color::Rgb(v, _) => v.is_nan().any(),
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
            Color::Rgb(v, cs) => Color::Rgb(
                v.clamp(Float3Expr::zero(), Float3Expr::splat(max_contrib)),
                *cs,
            ),
        }
    }
}
impl std::ops::Mul<Float> for &Color {
    type Output = Color;

    fn mul(self, rhs: Float) -> Self::Output {
        match self {
            Color::Spectral(s) => Color::Spectral(*s * rhs),
            Color::Rgb(s, cs) => Color::Rgb(*s * rhs, *cs),
        }
    }
}
impl std::ops::Div<Float> for &Color {
    type Output = Color;

    fn div(self, rhs: Float) -> Self::Output {
        match self {
            Color::Spectral(s) => Color::Spectral(*s / rhs),
            Color::Rgb(s, cs) => Color::Rgb(*s / rhs, *cs),
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
        assert_eq!(self.repr(), rhs.repr());
        match (self, rhs) {
            (Color::Spectral(s), Color::Spectral(t)) => Color::Spectral(*s * *t),
            (Color::Rgb(s, cs0), Color::Rgb(t, cs1)) => {
                assert_eq!(cs0, cs1);
                Color::Rgb(*s * *t, *cs0)
            }
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}
impl std::ops::Add<&Color> for &Color {
    type Output = Color;

    fn add(self, rhs: &Color) -> Self::Output {
        assert_eq!(self.repr(), rhs.repr());
        match (self, rhs) {
            (Color::Spectral(s), Color::Spectral(t)) => Color::Spectral(*s + *t),
            (Color::Rgb(s, cs0), Color::Rgb(t, cs1)) => {
                assert_eq!(cs0, cs1);
                Color::Rgb(*s + *t, *cs0)
            }
            _ => panic!("cannot multiply spectral and rgb"),
        }
    }
}
impl std::ops::Sub<&Color> for &Color {
    type Output = Color;

    fn sub(self, rhs: &Color) -> Self::Output {
        assert_eq!(self.repr(), rhs.repr());
        match (self, rhs) {
            (Color::Spectral(s), Color::Spectral(t)) => Color::Spectral(*s - *t),
            (Color::Rgb(s, cs0), Color::Rgb(t, cs1)) => {
                assert_eq!(cs0, cs1);
                Color::Rgb(*s - *t, *cs0)
            }
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

pub fn xyz_to_srgb_mat() -> glam::Mat3 {
    glam::Mat3::from_cols_array_2d(&[
        [3.240479, -1.537150, -0.498535],
        [-0.969256, 1.875991, 0.041556],
        [0.055648, -0.204043, 1.057311],
    ])
    .transpose()
}
pub fn srgb_to_xyz_mat() -> glam::Mat3 {
    glam::Mat3::from_cols_array_2d(&[
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ])
    .transpose()
}
pub fn srgb_to_aces_with_cat_mat() -> glam::Mat3 {
    glam::Mat3::from_cols_array_2d(&[
        [0.612494199, 0.338737252, 0.048855526],
        [0.070594252, 0.917671484, 0.011704306],
        [0.020727335, 0.106882232, 0.872338062],
    ])
    .transpose()
}
pub fn aces_to_srgb_with_cat_mat() -> glam::Mat3 {
    glam::Mat3::from_cols_array_2d(&[
        [1.707062673, -0.619959540, -0.087259850],
        [-0.130976829, 1.139032275, -0.007956297],
        [-0.024510601, -0.124810932, 1.149395971],
    ])
    .transpose()
}
pub fn xyz_to_aces_2065_1_mat() -> glam::Mat3 {
    glam::Mat3::from_cols_array_2d(&[
        [1.0498110175, 0.0000000000, -0.0000974845],
        [-0.4959030231, 1.3733130458, 0.0982400361],
        [0.0000000000, 0.0000000000, 0.9912520182],
    ])
    .transpose()
}
pub fn aces_2065_1_to_xyz_mat() -> glam::Mat3 {
    glam::Mat3::from_cols_array_2d(&[
        [0.9525523959, 0.0000000000, 0.0000936786],
        [0.3439664498, 0.7281660966, -0.0721325464],
        [0.0000000000, 0.0000000000, 1.0088251844],
    ])
    .transpose()
}

pub fn aces_2065_1_to_aces_cg_mat() -> glam::Mat3 {
    glam::Mat3::from_cols_array_2d(&[
        [1.4514393161, -0.2365107469, -0.2149285693],
        [-0.0765537733, 1.1762296998, -0.0996759265],
        [0.0083161484, -0.0060324498, 0.9977163014],
    ])
    .transpose()
}
pub fn aces_cg_to_aces_2065_1_mat() -> glam::Mat3 {
    glam::Mat3::from_cols_array_2d(&[
        [0.6954522414, 0.1406786965, 0.1638690622],
        [0.0447945634, 0.8596711184, 0.0955343182],
        [-0.0055258826, 0.0040252103, 1.0015006723],
    ])
    .transpose()
}
#[derive(Clone, Copy)]
pub struct ColorPipeline {
    pub color_repr: ColorRepr,
    pub rgb_colorspace: RgbColorSpace,
}
