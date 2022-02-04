use std::ops::{Index, IndexMut};

use crate::*;
#[derive(Clone, Copy, Debug)]
pub struct RGBSpectrum {
    pub samples: Vec3A,
}

pub fn srgb_to_linear(rgb: Vec3) -> Vec3 {
    let f = |s| -> f32 {
        if s <= 0.04045 {
            s / 12.92
        } else {
            (((s + 0.055) / 1.055) as f32).powf(2.4)
        }
    };
    vec3(f(rgb.x), f(rgb.y), f(rgb.z))
}
pub fn linear_to_srgb(linear: Vec3) -> Vec3 {
    let f = |l: f32| -> f32 {
        if l <= 0.0031308 {
            l * 12.92
        } else {
            l.powf(1.0 / 2.4) * 1.055 - 0.055
        }
    };

    vec3(f(linear.x), f(linear.y), f(linear.z))
}
impl RGBSpectrum {
    pub const N_SAMPLES: usize = 3;
    pub fn from_srgb(rgb: Vec3) -> RGBSpectrum {
        RGBSpectrum {
            samples: srgb_to_linear(rgb).into(),
        }
    }
    pub fn to_srgb(&self) -> Vec3 {
        linear_to_srgb(self.samples.into())
    }
    pub fn to_rgb_linear(&self) -> Vec3 {
        self.samples.into()
    }
    pub fn from_rgb_linear(rgb: Vec3) -> Self {
        Self {
            samples: rgb.into(),
        }
    }
    pub const fn zero() -> RGBSpectrum {
        Self {
            samples: Vec3A::ZERO,
        }
    }
    pub const fn one() -> Spectrum {
        Self {
            samples: Vec3A::ONE,
        }
    }
    // not necessarily black, but any value that is either black or invalid
    pub fn is_black(&self) -> bool {
        !self.samples.is_finite()
            || self.samples.cmpeq(Vec3A::ZERO).all()
            || self.samples.cmplt(Vec3A::ZERO).any()
        // self.samples.iter().any(|x| !x.is_finite())
        //     || self.samples.iter().all(|x| x == 0.0)
        //     || self.samples.iter().any(|x| x < 0.0)
    }
    pub fn lerp(x: RGBSpectrum, y: RGBSpectrum, a: f32) -> RGBSpectrum {
        x * (1.0 - a) + y * a
    }
}
pub fn lerp3v3(v0: Vec3, v1: Vec3, v2: Vec3, uv: Vec2) -> Vec3 {
    (1.0 - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2
}
pub fn lerp3v2(v0: Vec2, v1: Vec2, v2: Vec2, uv: Vec2) -> Vec2 {
    (1.0 - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2
}
pub fn lerp_scalar(x: f32, y: f32, a: f32) -> f32 {
    x + (y - x) * a
}
impl Index<usize> for RGBSpectrum {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.samples[index]
    }
}
impl IndexMut<usize> for RGBSpectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.samples[index]
    }
}
impl std::ops::Add for RGBSpectrum {
    type Output = RGBSpectrum;
    fn add(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples + rhs.samples,
        }
    }
}
impl std::ops::Sub for RGBSpectrum {
    type Output = RGBSpectrum;
    fn sub(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples - rhs.samples,
        }
    }
}
impl std::ops::AddAssign for RGBSpectrum {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl std::ops::MulAssign for RGBSpectrum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl std::ops::MulAssign<f32> for RGBSpectrum {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}
impl std::ops::Mul for Spectrum {
    type Output = Spectrum;
    fn mul(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples * rhs.samples,
        }
    }
}
impl std::ops::Mul<f32> for Spectrum {
    type Output = Spectrum;
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            samples: self.samples * rhs,
        }
    }
}
impl std::ops::Div<f32> for Spectrum {
    type Output = Spectrum;
    fn div(self, rhs: f32) -> Self::Output {
        Self {
            samples: self.samples / rhs,
        }
    }
}
pub type Spectrum = RGBSpectrum;
pub fn hsv_to_rgb(hsv: Vec3) -> Vec3 {
    let h = (hsv[0] / 60.0).floor() as u32;
    let f = hsv[0] / 60.0 - h as f32;
    let p = hsv[2] * (1.0 - hsv[1]);
    let q = hsv[2] * (1.0 - f * hsv[1]);
    let t = hsv[2] * (1.0 - (1.0 - f) * hsv[1]);
    let v = hsv[2];
    let (r, g, b) = match h {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => unreachable!(),
    };
    vec3(r, g, b)
}
pub fn rgb_to_hsl(rgb: Vec3) -> Vec3 {
    let max = rgb.max_element();
    let min = rgb.min_element();
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let h = {
        if max == min {
            0.0
        } else if max == r && g >= b {
            60.0 * (g - b) / (max - min)
        } else if max == r && g < b {
            60.0 * (g - b) / (max - min) + 360.0
        } else if max == g {
            60.0 * (b - r) / (max - min) + 120.0
        } else if max == b {
            60.0 * (r - g) / (max - min) + 240.0
        } else {
            unreachable!()
        }
    };
    let l = 0.5 * (max + min);
    let s = {
        if l == 0.0 || max == min {
            0.0
        } else if 0.0 < l || l <= 0.5 {
            (max - min) / (2.0 * l)
        } else if l > 0.5 {
            (max - min) / (2.0 - 2.0 * l)
        } else {
            unreachable!()
        }
    };
    vec3(h, s, l)
}

pub fn rgb_to_hsv(rgb: Vec3) -> Vec3 {
    let max = rgb.max_element();
    let min = rgb.min_element();
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let h = {
        if max == min {
            0.0
        } else if max == r && g >= b {
            60.0 * (g - b) / (max - min)
        } else if max == r && g < b {
            60.0 * (g - b) / (max - min) + 360.0
        } else if max == g {
            60.0 * (b - r) / (max - min) + 120.0
        } else if max == b {
            60.0 * (r - g) / (max - min) + 240.0
        } else {
            unreachable!()
        }
    };
    let v = max;
    let s = {
        if max == 0.0 {
            0.0
        } else {
            (max - min) / max
        }
    };
    vec3(h, s, v)
}
