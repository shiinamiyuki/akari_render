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
    /// | pixels | splat | weights |
    pub(crate) data: Buffer<f32>,
    repr: FilmColorRepr,
    resolution: Uint2,
    splat_scale: f32,
    filter: PixelFilter,
    copy_to_rgba_image: Option<Kernel<fn(Tex2d<Float4>, f32, bool)>>,
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
    pub fn splat_offset(&self) -> u32 {
        self.resolution.x * self.resolution.y * self.repr.nvalues() as u32
    }
    pub fn weight_offset(&self) -> u32 {
        self.splat_offset() + self.resolution.x * self.resolution.y * self.repr.nvalues() as u32
    }
    #[tracked]
    pub fn new(
        device: Device,
        resolution: Uint2,
        repr: FilmColorRepr,
        filter: PixelFilter,
    ) -> Self {
        let nvalues = repr.nvalues();
        let data = device.create_buffer::<f32>(
            resolution.x as usize * resolution.y as usize * (1 + 2 * nvalues),
        );
        let mut film = Self {
            device: device.clone(),
            data,
            repr,
            resolution,
            filter,
            splat_scale: 1.0,
            copy_to_rgba_image: None,
        };
        let copy_to_rgba_image = device.create_kernel_async::<fn(Tex2d<Float4>, f32, bool)>(
            &|image: Tex2dVar<Float4>, splat_scale: Expr<f32>, hdr: Expr<bool>| {
                let p = dispatch_id().xy();
                let i = p.x + p.y * resolution.x;
                let data = film.data.var();
                let nvalues = repr.nvalues() as u32;
                match repr {
                    FilmColorRepr::SRgb => {
                        let s_r = data.read(film.splat_offset() + i * nvalues + 0) * splat_scale;
                        let s_g = data.read(film.splat_offset() + i * nvalues + 1) * splat_scale;
                        let s_b = data.read(film.splat_offset() + i * nvalues + 2) * splat_scale;

                        let r = data.read(i * nvalues + 0);
                        let g = data.read(i * nvalues + 1);
                        let b = data.read(i * nvalues + 2);
                        let w = data.read(film.weight_offset() + i);
                        let rgb = Float3::expr(r, g, b) / select(w.eq(0.0), 1.0f32.expr(), w);
                        let rgb = rgb + Float3::expr(s_r, s_g, s_b);
                        let rgb = if_!(hdr, { rgb }, else, { linear_to_srgb(rgb) });
                        image.write(p, Float4::expr(rgb.x, rgb.y, rgb.z, 1.0f32));
                    }
                    _ => todo!(),
                }
            },
        );
        film.copy_to_rgba_image = Some(copy_to_rgba_image);
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
        let data = self.data.var();
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
                        data.atomic_fetch_add(
                            self.splat_offset() + i * nvalues as u32 + c as u32,
                            v,
                        );
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
        let data = self.data.var();
        let i = self.linear_index(p);
        let nvalues = self.repr.nvalues();
        match self.repr {
            FilmColorRepr::SRgb => {
                let rgb: Expr<Float3> = color.to_rgb(RgbColorSpace::SRgb);
                (0..nvalues).for_each(|c| {
                    let v = rgb[c as i32];
                    let j = i * nvalues as u32 + c as u32;
                    data.write(j, data.read(j) + v);
                });
                data.write(
                    self.weight_offset() + i,
                    data.read(self.weight_offset() + i) + weight,
                );
            }
            _ => todo!(),
        }
    }
    #[tracked]
    pub fn clear(&self) {
        self.data.view(..).fill(0.0);
    }

    pub fn copy_to_rgba_image(&self, image: &Tex2d<Float4>, hdr: bool) {
        assert_eq!(image.width(), self.resolution.x);
        assert_eq!(image.height(), self.resolution.y);

        self.copy_to_rgba_image.as_ref().unwrap().dispatch(
            [self.resolution.x, self.resolution.y, 1],
            image,
            &self.splat_scale,
            &hdr,
        );
    }
}
