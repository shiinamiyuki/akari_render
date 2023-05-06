use crate::{
    color::{Color, ColorRepr},
    *,
};

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
pub struct Film {
    device: Device,
    pixels: Buffer<f32>,
    weights: Buffer<f32>,
    repr: FilmColorRepr,
    resolution: Uint2,
}

impl Film {
    pub fn resolution(&self) -> Uint2 {
        self.resolution
    }
    pub fn new(device: Device, resolution: Uint2, color: FilmColorRepr) -> luisa::Result<Self> {
        let nvalues = color.nvalues();
        let pixels =
            device.create_buffer::<f32>(resolution.x as usize * resolution.y as usize * nvalues)?;
        let weights = device.create_buffer::<f32>(resolution.x as usize * resolution.y as usize)?;
        Ok(Self {
            device,
            pixels,
            weights,
            repr: color,
            resolution,
        })
    }
    fn linear_index(&self, p: Expr<Float2>) -> Expr<u32> {
        let resolution = const_(self.resolution);
        let ip = p.floor().int();
        let oob = ip.cmplt(0).any() | ip.cmpge(resolution.int()).any();
        lc_assert!(!oob);
        let ip = ip.uint();
        ip.x() + ip.y() * resolution.x()
    }
    pub fn add_sample(&self, p: Expr<Float2>, color: &Color, weight: Expr<f32>) {
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
    pub fn copy_to_rgba_image(&self, image: &Tex2d<Float4>) -> luisa::Result<()> {
        assert_eq!(image.width(), self.resolution.x);
        assert_eq!(image.height(), self.resolution.y);
        self.device
            .create_kernel::<()>(&|| {
                let p = dispatch_id().xy();
                let i = p.x() + p.y() * self.resolution.x;
                let pixels = self.pixels.var();
                let weights = self.weights.var();
                let nvalues = self.repr.nvalues();
                match self.repr {
                    FilmColorRepr::SRgb => {
                        let r = pixels.read(i * nvalues as u32 + 0);
                        let g = pixels.read(i * nvalues as u32 + 1);
                        let b = pixels.read(i * nvalues as u32 + 2);
                        let w = weights.read(i);
                        let rgb = make_float3(r, g, b) / select(w.cmpeq(0.0), const_(1.0f32), w);
                        image
                            .var()
                            .write(p, make_float4(rgb.x(), rgb.y(), rgb.z(), 1.0f32));
                    }
                    _ => todo!(),
                }
            })?
            .dispatch([self.resolution.x, self.resolution.y, 1])
    }
}
