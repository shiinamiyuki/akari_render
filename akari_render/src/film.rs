use std::f32::consts::PI;

use crate::{
    color::{linear_to_srgb, Color, RgbColorSpace, SampledWavelengths},
    util::safe_div,
    *,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(tag = "type")]
pub enum FilmColorRepr {
    #[serde(rename = "srgb")]
    SRgb,
    #[serde(rename = "xyz")]
    Xyz,
    #[serde(rename = "spectral")]
    Spectral { n: usize },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PixelFilter {
    #[serde(rename = "box")]
    Box,
    #[serde(rename = "gaussian")]
    Gaussian { radius: f32 },
}
impl PixelFilter {
    #[tracked]
    pub fn sample(&self, u: Expr<Float2>) -> (Expr<Float2>, Expr<f32>) {
        match self {
            PixelFilter::Box => (u * 0.5 - 0.5, 1.0f32.expr()),
            PixelFilter::Gaussian { radius: width } => {
                let sigma = *width / 3.0;
                let u1 = u.x;
                let u2 = u.y;
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * PI * u2;
                let offset = Float2::expr(r * theta.cos(), r * theta.sin()) * sigma;
                let offset = offset.clamp(
                    Float2::splat_expr(-width.expr()),
                    Float2::splat_expr(width.expr()),
                );
                (offset, 1.0f32.expr())
            }
        }
    }
}
impl Default for PixelFilter {
    fn default() -> Self {
        PixelFilter::Gaussian { radius: 1.5 }
    }
}
impl FilmColorRepr {
    pub fn nvalues(&self) -> usize {
        match self {
            FilmColorRepr::SRgb => 3,
            FilmColorRepr::Xyz => 3,
            FilmColorRepr::Spectral { n } => *n,
        }
    }
}

pub struct Film {
    device: Device,
    pub(crate) pixels: Buffer<f32>,
    pub(crate) weights: Buffer<f32>,
    pub(crate) splat: Buffer<f32>,
    repr: FilmColorRepr,
    resolution: Uint2,
    splat_scale: f32,
    filter: PixelFilter,
    copy_to_rgba_image: Kernel<fn(Tex2d<Float4>, f32, bool)>,
}

impl Film {
    pub fn nchannels(&self) -> usize {
        self.repr.nvalues()
    }
    pub fn resolution(&self) -> Uint2 {
        self.resolution
    }
    pub fn filter(&self) -> PixelFilter {
        self.filter
    }
    #[tracked]
    pub fn new(
        device: Device,
        resolution: Uint2,
        repr: FilmColorRepr,
        filter: PixelFilter,
    ) -> Self {
        let nvalues = repr.nvalues();
        let pixels =
            device.create_buffer::<f32>(resolution.x as usize * resolution.y as usize * nvalues);
        let splat =
            device.create_buffer::<f32>(resolution.x as usize * resolution.y as usize * nvalues);
        let weights = device.create_buffer::<f32>(resolution.x as usize * resolution.y as usize);
        let copy_to_rgba_image = Kernel::<fn(Tex2d<Float4>, f32, bool)>::new(
            &device,
            &|image: Tex2dVar<Float4>, splat_scale: Expr<f32>, hdr: Expr<bool>| {
                let p = dispatch_id().xy();
                let i = p.x + p.y * resolution.x;
                let pixels = pixels.var();
                let splat = splat.var();
                let weights = weights.var();
                let nvalues = repr.nvalues();
                match repr {
                    FilmColorRepr::SRgb => {
                        let s_r = splat.read(i * nvalues as u32 + 0) * splat_scale;
                        let s_g = splat.read(i * nvalues as u32 + 1) * splat_scale;
                        let s_b = splat.read(i * nvalues as u32 + 2) * splat_scale;

                        let r = pixels.read(i * nvalues as u32 + 0);
                        let g = pixels.read(i * nvalues as u32 + 1);
                        let b = pixels.read(i * nvalues as u32 + 2);
                        let w = weights.read(i);
                        let rgb = Float3::expr(r, g, b) / select(w.eq(0.0), 1.0f32.expr(), w);
                        let rgb = rgb + Float3::expr(s_r, s_g, s_b);
                        let rgb = if_!(hdr, { rgb }, else, { linear_to_srgb(rgb) });
                        image.write(p, Float4::expr(rgb.x, rgb.y, rgb.z, 1.0f32));
                    }
                    _ => todo!(),
                }
            },
        );
        let film = Self {
            device,
            splat,
            pixels,
            weights,
            repr,
            resolution,
            filter,
            splat_scale: 1.0,
            copy_to_rgba_image,
        };
        film.clear();
        film
    }
    pub fn set_splat_scale(&mut self, scale: f32) {
        self.splat_scale = scale;
    }
    #[tracked]
    fn linear_index(&self, p: Expr<Float2>) -> Expr<u32> {
        let resolution = self.resolution.expr();
        let ip = p.floor().cast_i32();
        let oob = ip.lt(0).any() | ip.ge(resolution.cast_i32()).any();
        if debug_mode() {
            lc_assert!(!oob);
        }
        let ip = ip.cast_u32();
        ip.x + ip.y * resolution.x
    }
    pub fn add_splat(
        &self,
        p: Expr<Float2>,
        color: &Color,
        _swl: Expr<SampledWavelengths>,
        weight: Expr<f32>,
    ) {
        let splat = self.splat.var();
        let i = self.linear_index(p);
        let nvalues = self.repr.nvalues();
        let color = color.remove_nan() * weight;
        match self.repr {
            FilmColorRepr::SRgb => {
                let rgb: Expr<Float3> = color.to_rgb(RgbColorSpace::SRgb);
                for c in 0..nvalues {
                    track!({
                        let v = rgb[c as i32];
                        let v = select(v.is_nan(), 0.0f32.expr(), v);
                        splat.atomic_fetch_add(i * nvalues as u32 + c as u32, v);
                    });
                }
            }
            _ => todo!(),
        }
    }
    #[tracked]
    pub fn add_sample(
        &self,
        p: Expr<Float2>,
        color: &Color,
        _swl: Expr<SampledWavelengths>,
        weight: Expr<f32>,
    ) {
        let color = color.remove_nan() * weight;
        let pixels = self.pixels.var();
        let weights = self.weights.var();
        let i = self.linear_index(p);
        let nvalues = self.repr.nvalues();
        match self.repr {
            FilmColorRepr::SRgb => {
                let rgb: Expr<Float3> = color.to_rgb(RgbColorSpace::SRgb);
                (0..nvalues).for_each(|c| {
                    let v = rgb[c as i32];
                    let j = i * nvalues as u32 + c as u32;
                    pixels.write(j, pixels.read(j) + v);
                });
                weights.write(i, weights.read(i) + weight);
            }
            _ => todo!(),
        }
    }
    #[tracked]
    pub fn clear(&self) {
        self.pixels.view(..).fill(0.0);
        self.weights.view(..).fill(0.0);
    }
    /// merge the content of `other` into `self`
    /// `self` and `other` must have the same resolution
    /// The splat buffer of `other` will be converted using `self.splat_scale`
    #[tracked]
    pub fn merge(&self, other: &Self) {
        assert_eq!(self.repr, other.repr);
        assert_eq!(self.resolution, other.resolution);
        Kernel::<fn(f32, f32)>::new(
            &self.device,
            &|self_splat_scale: Expr<f32>, other_splat_scale: Expr<f32>| {
                let p = dispatch_id().xy();
                let i = p.x + p.y * self.resolution.x;
                let pixels = self.pixels.var();
                let splat = self.splat.var();
                let weights = self.weights.var();
                let nvalues = self.repr.nvalues();
                let w = other.weights.var().read(i);
                weights.atomic_fetch_add(i, w);
                escape!({
                    for c in 0..nvalues {
                        track!({
                            // merge splat
                            let s = safe_div(
                                other.splat.var().read(i * nvalues as u32 + c as u32),
                                self_splat_scale,
                            ) * other_splat_scale;
                            splat.atomic_fetch_add(i * nvalues as u32 + c as u32, s);

                            // merge pixels
                            let p = other.pixels.var().read(i * nvalues as u32 + c as u32);
                            pixels.atomic_fetch_add(i * nvalues as u32 + c as u32, p * w);
                        });
                    }
                });
            },
        )
        .dispatch(
            [self.resolution.x, self.resolution.y, 1],
            &self.splat_scale,
            &other.splat_scale,
        );
    }
    pub fn copy_to_rgba_image(&self, image: &Tex2d<Float4>, hdr: bool) {
        assert_eq!(image.width(), self.resolution.x);
        assert_eq!(image.height(), self.resolution.y);

        self.copy_to_rgba_image.dispatch(
            [self.resolution.x, self.resolution.y, 1],
            image,
            &self.splat_scale,
            &hdr,
        );
    }
}
