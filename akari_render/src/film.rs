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
    pub fn sample(&self, u: Expr<Float2>) -> (Expr<Float2>, Expr<f32>) {
        match self {
            PixelFilter::Box => (u * 0.5 - 0.5, const_(1.0f32)),
            PixelFilter::Gaussian { radius: width } => {
                let sigma = *width / 3.0;
                let u1 = u.x();
                let u2 = u.y();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * PI * u2;
                let offset = make_float2(r * theta.cos(), r * theta.sin()) * sigma;
                let offset = offset.clamp(-*width, *width);
                (offset, const_(1.0f32))
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
        let copy_to_rgba_image = device.create_kernel::<fn(Tex2d<Float4>, f32, bool)>(
            &|image: Tex2dVar<Float4>, splat_scale: Expr<f32>, hdr: Expr<bool>| {
                let p = dispatch_id().xy();
                let i = p.x() + p.y() * resolution.x;
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
                        let rgb = make_float3(r, g, b) / select(w.cmpeq(0.0), const_(1.0f32), w);
                        let rgb = rgb + make_float3(s_r, s_g, s_b);
                        let rgb = if_!(hdr, { rgb }, else, { linear_to_srgb(rgb) });
                        image.write(p, make_float4(rgb.x(), rgb.y(), rgb.z(), 1.0f32));
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
    fn linear_index(&self, p: Expr<Float2>) -> Expr<u32> {
        let resolution = const_(self.resolution);
        let ip = p.floor().int();
        let oob = ip.cmplt(0).any() | ip.cmpge(resolution.int()).any();
        lc_assert!(!oob);
        let ip = ip.uint();
        ip.x() + ip.y() * resolution.x()
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
                let rgb: Float3Expr = color.to_rgb(RgbColorSpace::SRgb);
                for c in 0..nvalues {
                    let v = rgb.at(c);
                    let v = select(v.is_nan(), 0.0.into(), v);
                    splat.atomic_fetch_add(i * nvalues as u32 + c as u32, v);
                }
            }
            _ => todo!(),
        }
    }
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
                let rgb: Float3Expr = color.to_rgb(RgbColorSpace::SRgb);
                for c in 0..nvalues {
                    let v = rgb.at(c);
                    let v = select(v.is_nan(), 0.0.into(), v);
                    pixels.atomic_fetch_add(i * nvalues as u32 + c as u32, v);
                }
                weights.atomic_fetch_add(i, weight);
            }
            _ => todo!(),
        }
    }
    pub fn clear(&self) {
        self.pixels.view(..).fill(0.0);
        self.weights.view(..).fill(0.0);
    }
    /// merge the content of `other` into `self`
    /// `self` and `other` must have the same resolution
    /// The splat buffer of `other` will be converted using `self.splat_scale`
    pub fn merge(&self, other: &Self) {
        assert_eq!(self.repr, other.repr);
        assert_eq!(self.resolution, other.resolution);
        self.device
            .create_kernel::<fn(f32, f32)>(
                &|self_splat_scale: Expr<f32>, other_splat_scale: Expr<f32>| {
                    let p = dispatch_id().xy();
                    let i = p.x() + p.y() * self.resolution.x;
                    let pixels = self.pixels.var();
                    let splat = self.splat.var();
                    let weights = self.weights.var();
                    let nvalues = self.repr.nvalues();
                    let w = other.weights.var().read(i);
                    weights.atomic_fetch_add(i, w);
                    for c in 0..nvalues {
                        // merge splat
                        let s = safe_div(
                            other.splat.var().read(i * nvalues as u32 + c as u32),
                            self_splat_scale,
                        ) * other_splat_scale;
                        splat.atomic_fetch_add(i * nvalues as u32 + c as u32, s);

                        // merge pixels
                        let p = other.pixels.var().read(i * nvalues as u32 + c as u32);
                        pixels.atomic_fetch_add(i * nvalues as u32 + c as u32, p * w);
                    }
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

#[derive(Clone, Copy, Value, Debug)]
#[repr(C)]
struct VarTracker {
    mean: f32,
    s: f32,
    w_sum: f32,
    w2_sum: f32,
}

#[derive(Clone)]
pub struct VarFilm {
    #[allow(dead_code)]
    device: Device,
    locks: Buffer<u32>,
    vars: Buffer<VarTracker>,
    resolution: Uint2,
}

impl VarFilm {
    pub fn new(device: Device, resolution: Uint2) -> Self {
        let vars =
            device.create_buffer::<VarTracker>(resolution.x as usize * resolution.y as usize);
        let locks = device.create_buffer::<u32>(resolution.x as usize * resolution.y as usize);
        Self {
            device,
            vars,
            resolution,
            locks,
        }
    }
    fn linear_index(&self, p: Expr<Uint2>) -> Expr<u32> {
        let resolution = const_(self.resolution);
        let ip = p.int();
        let oob = ip.cmplt(0).any() | ip.cmpge(resolution.int()).any();
        lc_assert!(!oob);
        let ip = ip.uint();
        ip.x() + ip.y() * resolution.x()
    }
    pub fn add_sample(&self, p: Expr<Uint2>, v: Expr<f32>, w: Expr<f32>) {
        let i = self.linear_index(p);
        let locks = self.locks.var();
        while_!(locks.atomic_compare_exchange(i, 0, 1).cmpne(0), {});
        let var = self.vars.var().read(i);
        let var = var!(VarTracker, var);
        var.set_w_sum(var.w_sum().load() + w);
        var.set_w2_sum(var.w2_sum().load() + w * w);
        let mean_old = var.mean().load();
        var.set_mean(mean_old + (w / var.w_sum().load()) * (v - mean_old));
        var.set_s(var.s().load() + w * (v - mean_old) * (v - var.mean().load()));
        self.vars.var().write(i, var.load());
        locks.write(i, 0);
    }
    pub fn mean(&self, p: Expr<Uint2>) -> Expr<f32> {
        let i = self.linear_index(p);
        self.vars.var().read(i).mean()
    }
    pub fn variance(&self, p: Expr<Uint2>) -> Expr<f32> {
        let i = self.linear_index(p);
        let var = self.vars.var().read(i);
        safe_div(var.s(), var.w_sum())
    }
    pub fn unbiased_variance(&self, p: Expr<Uint2>) -> Expr<f32> {
        let i = self.linear_index(p);
        let var = self.vars.var().read(i);
        safe_div(var.s(), var.w_sum() - 1.0)
    }
}
