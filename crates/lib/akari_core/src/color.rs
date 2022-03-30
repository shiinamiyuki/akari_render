use crate::{
    rgb2spec::{Rgb2SpectrumData, SPECTRUM_TABLE_RES},
    *,
};
use akari_common::glam::mat3;
use serde::{Deserialize, Serialize};
pub use util::{hsv_to_rgb, linear_to_srgb, rgb_to_hsl, rgb_to_hsv, srgb_to_linear};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct XYZ {
    values: Vec3,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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
    let m = Mat3::from_cols_array(&m);
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
            0.5f32 + x / (2.0 * (1.0 + x * x).sqrt())
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
    pub fn sample(&self, swl: &SampledWavelengths) -> SampledSpectrum {
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
#[derive(Clone, Copy)]
pub struct RgbColorSpace {
    id: RgbColorSpaceId,
    rgb2spec_data: &'static Rgb2SpectrumData,
    illuminant: &'static dyn Spectrum,
}
impl RgbColorSpace {
    pub fn new(id: RgbColorSpaceId) -> Self {
        match id {
            RgbColorSpaceId::SRgb => Self {
                id,
                rgb2spec_data: &rgb2spec::srgb::DATA,
                illuminant: spectrum_from_name("stdillum-D65"),
            },
            RgbColorSpaceId::Aces2065_1 => Self {
                id,
                rgb2spec_data: &rgb2spec::aces2065_1::DATA,
                illuminant: spectrum_from_name("stdillum-D65"),
            },
            RgbColorSpaceId::Rec2020 => Self {
                id,
                rgb2spec_data: &rgb2spec::rec2020::DATA,
                illuminant: spectrum_from_name("stdillum-D65"),
            },
            RgbColorSpaceId::DCIP3 => Self {
                id,
                rgb2spec_data: &rgb2spec::dci_p3::DATA,
                illuminant: spectrum_from_name("stdillum-D65"),
            },
        }
    }
    pub fn illuminant(&self) -> &'static dyn Spectrum {
        self.illuminant
    }
    pub fn rgb2spec(&self, rgb: Vec3) -> RgbSigmoidPolynomial {
        debug_assert!(rgb.max_element() <= 1.0);
        debug_assert!(rgb.min_element() >= 0.0);
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
        let y = rgb[(maxc + 2) % 3] / z * (SPECTRUM_TABLE_RES - 1) as f32;

        let znodes = &self.rgb2spec_data.scale;

        let xi = (x as usize).min(SPECTRUM_TABLE_RES - 2);
        let yi = (y as usize).min(SPECTRUM_TABLE_RES - 2);
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
