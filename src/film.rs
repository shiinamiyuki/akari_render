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
    pixels: Buffer<f32>,
    weights: Buffer<f32>,
    repr: FilmColorRepr,
    resolution: Uint2,
}

impl Film {
    pub fn new(device: Device, resolution: Uint2, color: FilmColorRepr) -> Self {
        let nvalues = color.nvalues();
        let pixels = device
            .create_buffer::<f32>(resolution.x as usize * resolution.y as usize * nvalues)
            .unwrap();
        let weights = device
            .create_buffer::<f32>(resolution.x as usize * resolution.y as usize)
            .unwrap();
        Self {
            pixels,
            weights,
            repr: color,
            resolution,
        }
    }
    pub fn add_sample(&self, p: Expr<Float2>, color: &Color, color_repr: &ColorRepr) {
        let pixels = self.pixels.var();
        let weights = self.weights.var();

        match color_repr {
            ColorRepr::Rgb => match self.repr {
                FilmColorRepr::SRgb => {}
                _ => todo!(),
            },
            _ => todo!(),
        }
    }
    pub fn clear(&self) {
        todo!()
    }
    pub fn copy_to_rgba_image(&self, image: &Tex2d<Float4>) {
        todo!()
    }
}
