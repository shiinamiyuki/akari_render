use akari_common::glam::mat3;

use crate::{
    rgb2spec::{Rgb2SpectrumData, SPECTRUM_TABLE_RES},
    *,
};

#[derive(Clone, Copy, Debug)]
pub struct XYZ {
    values: Vec3,
}
#[derive(Clone, Copy, Debug)]
pub struct SRgb {
    values: Vec3,
}
impl_color_like!(XYZ, Vec3);
impl_color_like!(SRgb, Vec3);
impl From<XYZ> for SRgb {
    fn from(xyz: XYZ) -> Self {
        Self::new(xyz_to_srgb(xyz.values()))
    }
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

pub fn xyz_to_srgb(xyz: Vec3) -> Vec3 {
    let m = mat3(
        vec3(3.240479f32, -0.969256f32, 0.055648f32),
        vec3(-1.537150f32, 1.875991f32, -0.204043f32),
        vec3(-0.498535f32, 0.041556f32, 1.057311f32),
    );
    m * xyz
}
pub fn srgb_to_xyz(srgb: Vec3) -> Vec3 {
    // let m = [
    //     [0.412453, 0.357580, 0.180423],
    //     [0.212671, 0.715160, 0.072169],
    //     [0.019334, 0.119193, 0.950227],
    // ];
    let m = [
        0.412453, 0.212671, 0.019334, //.
        0.357580, 0.715160, 0.119193, //.
        0.180423, 0.072169, 0.950227,
    ];
    let m = Mat3::from_cols_array(&m).transpose();
    m * srgb
}

#[macro_export]
macro_rules! impl_color_like {
    ($t:ty,$inner:ty) => {
        impl $t {
            pub fn new(values: $inner) -> Self {
                Self { values }
            }
            pub fn splat(value: f32) -> Self {
                Self {
                    values: <$inner>::splat(value),
                }
            }
            pub fn max_element(&self) -> f32 {
                self.values.max_element()
            }

            pub const fn zero() -> $t {
                Self {
                    values: <$inner>::ZERO,
                }
            }
            pub const fn one() -> $t {
                Self {
                    values: <$inner>::ONE,
                }
            }
            pub fn is_black(&self) -> bool {
                !self.values.is_finite()
                    || self.values.cmpeq(<$inner>::ZERO).all()
                    || self.values.cmplt(<$inner>::ZERO).any()
            }
            pub fn lerp(x: Self, y: Self, a: f32) -> Self {
                x * (1.0 - a) + y * a
            }
            pub const fn values(&self) -> $inner {
                self.values
            }
        }
        impl std::ops::Index<usize> for $t {
            type Output = f32;
            fn index(&self, index: usize) -> &Self::Output {
                &self.values[index]
            }
        }
        impl std::ops::IndexMut<usize> for $t {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.values[index]
            }
        }
        impl std::ops::Add for $t {
            type Output = $t;
            fn add(self, rhs: $t) -> Self::Output {
                Self {
                    values: self.values + rhs.values,
                }
            }
        }
        impl std::ops::Sub for $t {
            type Output = $t;
            fn sub(self, rhs: $t) -> Self::Output {
                Self {
                    values: self.values - rhs.values,
                }
            }
        }
        impl std::ops::AddAssign for $t {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl std::ops::MulAssign for $t {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl std::ops::MulAssign<f32> for $t {
            fn mul_assign(&mut self, rhs: f32) {
                *self = *self * rhs;
            }
        }
        impl std::ops::Mul for $t {
            type Output = $t;
            fn mul(self, rhs: $t) -> Self::Output {
                Self {
                    values: self.values * rhs.values,
                }
            }
        }
        impl std::ops::Mul<f32> for $t {
            type Output = $t;
            fn mul(self, rhs: f32) -> Self::Output {
                Self {
                    values: self.values * rhs,
                }
            }
        }
        impl std::ops::Div<f32> for $t {
            type Output = $t;
            fn div(self, rhs: f32) -> Self::Output {
                Self {
                    values: self.values / rhs,
                }
            }
        }
        impl std::ops::Div<$inner> for $t {
            type Output = $t;
            fn div(self, rhs: $inner) -> Self::Output {
                Self {
                    values: self.values / rhs,
                }
            }
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub struct RgbSigmoidPolynomial {
    c0: f32,
    c1: f32,
    c2: f32,
}

impl RgbSigmoidPolynomial {
    fn s(x: f32) -> f32 {
        if x.is_infinite() {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            0.5f32 + x / (2.0 + (1.0 + x * x).sqrt())
        }
    }
    pub fn new(c0: f32, c1: f32, c2: f32) -> Self {
        Self { c0, c1, c2 }
    }
    pub fn evaluate(&self, lambda: f32) -> f32 {
        Self::s(((self.c0 * lambda) + self.c1) * lambda + self.c2)
    }
    pub fn max_element(&self) -> f32 {
        let result = self.evaluate(360.0).max(self.evaluate(830.0));
        let lambda = -self.c1 / (2.0 * self.c0);
        if lambda >= 360.0 && lambda <= 830.0 {
            result.max(self.evaluate(lambda))
        } else {
            result
        }
    }
    pub fn sample(&self, swl: SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::new(vec4(
            self.evaluate(swl[0]),
            self.evaluate(swl[1]),
            self.evaluate(swl[2]),
            self.evaluate(swl[3]),
        ))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RgbColorSpaceId {
    SRgb,
    Aces2065_1,
    Rec2020,
    DCIP3,
}
pub struct RgbColorSpace {
    id: RgbColorSpaceId,
    rgb2spec_data: &'static Rgb2SpectrumData,
    illuminant: &'static dyn Spectrum,
}
impl RgbColorSpace {
    pub fn new(id: RgbColorSpaceId) -> Self {
        match id {
            RgbColorSpaceId::SRgb => {
                let data: &'static Rgb2SpectrumData = &rgb2spec::srgb::DATA;
                Self {
                    id,
                    rgb2spec_data: data,
                    illuminant: spectrum_from_name("stdillum-D65"),
                }
            }
            RgbColorSpaceId::Aces2065_1 => todo!(),
            RgbColorSpaceId::Rec2020 => todo!(),
            RgbColorSpaceId::DCIP3 => todo!(),
        }
    }
    pub fn rgb2spec(&self, rgb: Vec3) -> RgbSigmoidPolynomial {
        if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
            return RgbSigmoidPolynomial::new(
                0.0,
                0.0,
                (rgb[0] - 0.5) / (rgb[0] * (1.0 - rgb[0])).sqrt(),
            );
        }
        let maxc: usize = if rgb[0] > rgb[1] {
            if rgb[0] > rgb[2] {
                0
            } else {
                2
            }
        } else {
            if rgb[1] > rgb[2] {
                1
            } else {
                2
            }
        };
        let z = rgb[maxc];
        let x = rgb[(maxc + 1) % 3] / z * (SPECTRUM_TABLE_RES - 1) as f32;
        let y = rgb[(maxc + 1) % 3] / z * (SPECTRUM_TABLE_RES - 1) as f32;

        let znodes = &self.rgb2spec_data.scale;

        let xi = (x as usize).min(SPECTRUM_TABLE_RES - 1);
        let yi = (y as usize).min(SPECTRUM_TABLE_RES - 1);
        let zi = find_largest(znodes, |s| *s < z);

        let dx = x - xi as f32;
        let dy = y - yi as f32;
        let dz = (z - znodes[zi]) / (znodes[zi + 1] - znodes[zi]);

        let mut c = [0.0f32; 3];
        for i in 0..3usize {
            let co = |dx: usize, dy: usize, dz: usize| -> f32 {
                self.rgb2spec_data.table[maxc][zi + dz][yi + dy][xi + dx][i]
            };
            c[i] = trilinear(co, vec3(dx, dy, dz));
        }
        RgbSigmoidPolynomial::new(c[0], c[1], c[2])
    }
    pub fn id(&self) -> RgbColorSpaceId {
        self.id
    }
}
