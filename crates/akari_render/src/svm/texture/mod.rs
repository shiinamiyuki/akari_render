use crate::color::{
    aces_to_srgb_with_cat_mat, srgb_to_aces_with_cat_mat, srgb_to_linear, Color, RgbColorSpace,
    SampledWavelengths,
};

pub mod noise;
use super::*;

pub fn rgb_to_target_colorspace(
    rgb: Expr<Float3>,
    colorspace: RgbColorSpace,
    target: RgbColorSpace,
) -> Expr<Float3> {
    match colorspace {
        RgbColorSpace::SRgb => {
            if target == RgbColorSpace::ACEScg {
                track!(Mat3::from(srgb_to_aces_with_cat_mat()).expr() * rgb)
            } else {
                rgb
            }
        }
        RgbColorSpace::ACEScg => {
            if target == RgbColorSpace::SRgb {
                track!(Mat3::from(aces_to_srgb_with_cat_mat()).expr() * rgb)
            } else {
                rgb
            }
        }
    }
}
pub fn spectral_uplift(
    rgb: Expr<Float3>,
    colorspace: RgbColorSpace,
    _swl: Expr<SampledWavelengths>,
    color_repr: ColorRepr,
) -> Color {
    match color_repr {
        ColorRepr::Rgb(cs) => Color::Rgb(rgb_to_target_colorspace(rgb, colorspace, cs), cs),
        ColorRepr::Spectral => {
            todo!()
        }
    }
}
pub fn rgb_gamma_correction(rgb: Expr<Float3>, colorspace: RgbColorSpace) -> Expr<Float3> {
    match colorspace {
        RgbColorSpace::SRgb => srgb_to_linear(rgb),
        RgbColorSpace::ACEScg => {
            todo!()
        }
    }
}
