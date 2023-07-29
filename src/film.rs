use crate::{color::Color, util::safe_div, *};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FilmColorRepr {
    SRgb,
    Xyz,
    Spectral(usize),
}
impl FilmColorRepr {
    pub fn nvalues(&self) -> usize {
        match self {
            FilmColorRepr::SRgb => 3,
            FilmColorRepr::Xyz => 3,
            FilmColorRepr::Spectral(n) => *n,
        }
    }
}
#[derive(Clone)]
pub struct Film {
    device: Device,
    pub(crate) pixels: Buffer<f32>,
    pub(crate) weights: Buffer<f32>,
    pub(crate) splat: Buffer<f32>,
    repr: FilmColorRepr,
    resolution: Uint2,
    splat_scale: f32,
}

impl Film {
    pub fn nchannels(&self) -> usize {
        self.repr.nvalues()
    }
    pub fn resolution(&self) -> Uint2 {
        self.resolution
    }
    pub fn new(device: Device, resolution: Uint2, color: FilmColorRepr) -> Self {
        let nvalues = color.nvalues();
        let pixels =
            device.create_buffer::<f32>(resolution.x as usize * resolution.y as usize * nvalues);
        let splat =
            device.create_buffer::<f32>(resolution.x as usize * resolution.y as usize * nvalues);
        let weights = device.create_buffer::<f32>(resolution.x as usize * resolution.y as usize);
        Self {
            device,
            splat,
            pixels,
            weights,
            repr: color,
            resolution,
            splat_scale: 1.0,
        }
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
    pub fn add_splat(&self, p: Expr<Float2>, color: &Color, weight: Expr<f32>) {
        let splat = self.splat.var();
        let i = self.linear_index(p);
        let nvalues = self.repr.nvalues();
        let color = color * weight;
        match self.repr {
            FilmColorRepr::SRgb => {
                let rgb: Float3Expr = color.to_rgb();
                for c in 0..nvalues {
                    let v = rgb.at(c);
                    let v = select(v.is_nan(), 0.0.into(), v);
                    splat.atomic_fetch_add(i * nvalues as u32 + c as u32, v);
                }
            }
            _ => todo!(),
        }
    }
    pub fn add_sample(&self, p: Expr<Float2>, color: &Color, weight: Expr<f32>) {
        let color = color * weight;
        let pixels = self.pixels.var();
        let weights = self.weights.var();
        let i = self.linear_index(p);
        let nvalues = self.repr.nvalues();
        match self.repr {
            FilmColorRepr::SRgb => {
                let rgb: Float3Expr = color.to_rgb();
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
    pub fn copy_to_rgba_image(&self, image: &Tex2d<Float4>) {
        assert_eq!(image.width(), self.resolution.x);
        assert_eq!(image.height(), self.resolution.y);
        self.device
            .create_kernel::<(f32,)>(&|splat_scale: Expr<f32>| {
                let p = dispatch_id().xy();
                let i = p.x() + p.y() * self.resolution.x;
                let pixels = self.pixels.var();
                let splat = self.splat.var();
                let weights = self.weights.var();
                let nvalues = self.repr.nvalues();
                match self.repr {
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
                        image
                            .var()
                            .write(p, make_float4(rgb.x(), rgb.y(), rgb.z(), 1.0f32));
                    }
                    _ => todo!(),
                }
            })
            .dispatch([self.resolution.x, self.resolution.y, 1], &self.splat_scale)
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
