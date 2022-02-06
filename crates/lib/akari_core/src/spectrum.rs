use std::{collections::HashMap, sync::Arc};

pub use crate::color::*;
use crate::{
    function::{Dense1D, DenseSlice1D, Function1D, PiecewiseLinear1D},
    *,
};
use akari_const::{CIE_LAMBDA_MAX, CIE_LAMBDA_MIN, CIE_SAMPLES, CIE_Y};
use lazy_static::lazy_static;
pub const SPECTRUM_SAMPLES: usize = 4;
#[derive(Clone, Copy, Debug, Default)]
pub struct SampledSpectrum {
    values: Vec4,
}

impl_color_like!(SampledSpectrum, Vec4);
#[allow(dead_code)]
fn cie_y_integral() -> f32 {
    let mut s = 0.0;
    let mut i = 0;
    while i < CIE_SAMPLES - 1 {
        s += 0.5 * (CIE_Y[i] + CIE_Y[i + 1]);
        i += 1;
    }
    s
}
mod test {

    #[test]
    fn test_cie_y_integral() {
        use super::*;
        use akari_common::statrs::assert_almost_eq;

        use crate::spectrum::cie_y_integral;
        assert_almost_eq!(cie_y_integral() as f64, CIE_Y_INTEGRAL as f64, 0.001);
    }
}
pub const CIE_Y_INTEGRAL: f32 = 106.85694885253906;
pub const INV_CIE_Y_INTEGRAL: f32 = 1.0 / CIE_Y_INTEGRAL;
#[derive(Clone, Copy, Debug)]
pub struct SampledWavelengths {
    lambda: Vec4,
    pdf: Vec4,
}

impl SampledWavelengths {
    pub fn pdf(&self) -> SampledSpectrum {
        SampledSpectrum::new(self.pdf)
    }
    pub fn cie_xyz(&self, s: SampledSpectrum) -> XYZ {
        let x = &DenselySampledSpectrum2::CIE_X;
        let y = &DenselySampledSpectrum2::CIE_Y;
        let z = &DenselySampledSpectrum2::CIE_Z;
        let avg = |s: SampledSpectrum| (s[0] + s[1] + s[2] + s[3]) / SPECTRUM_SAMPLES as f32;
        let pdf = Vec4::select(self.pdf.cmple(Vec4::ZERO), Vec4::ONE, self.pdf);
        XYZ::new(vec3(
            avg((s * x.sample(*self)) / pdf),
            avg((s * y.sample(*self)) / pdf),
            avg((s * z.sample(*self)) / pdf),
        )) * INV_CIE_Y_INTEGRAL
    }
    pub fn sample_visible(u: f32) -> Self {
        let mut w = Self {
            lambda: Vec4::ZERO,
            pdf: Vec4::ZERO,
        };
        for i in 0..SPECTRUM_SAMPLES {
            let up = (u + i as f32 / SPECTRUM_SAMPLES as f32).fract();
            w.lambda[i] = sample_visible_wavelenghts(up);
            w.pdf[i] = visible_wavelenghts_pdf(w.lambda[i]);
        }
        w
    }
    pub fn sample_uniform(u: f32) -> Self {
        let mut w = Self {
            lambda: Vec4::ZERO,
            pdf: Vec4::ZERO,
        };
        for i in 0..SPECTRUM_SAMPLES {
            let up = (u + i as f32 / SPECTRUM_SAMPLES as f32).fract();
            w.lambda[i] = lerp(CIE_LAMBDA_MIN, CIE_LAMBDA_MAX, up);
            w.pdf[i] = 1.0 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN);
        }
        w
    }
}
impl std::ops::Index<usize> for SampledWavelengths {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.lambda[index]
    }
}
#[derive(Clone, Copy, Debug)]
pub struct RgbUnboundedSpectrum {
    spd: RgbSigmoidPolynomial,
    scale: f32,
}
impl Spectrum for RgbUnboundedSpectrum {
    fn sample(&self, swl: SampledWavelengths) -> SampledSpectrum {
        self.spd.sample(swl) * self.scale
    }
}
#[derive(Clone, Copy, Debug)]
pub struct RgbAlbedoSpectrum {
    spd: RgbSigmoidPolynomial,
}

impl Spectrum for RgbAlbedoSpectrum {
    fn sample(&self, swl: SampledWavelengths) -> SampledSpectrum {
        self.spd.sample(swl)
    }
}
pub struct RgbIlluminantSpectrum {
    spd: RgbSigmoidPolynomial,
    scale: f32,
    illuminant: &'static dyn Spectrum,
}
impl RgbIlluminantSpectrum {
    pub fn new(rep: RgbSigmoidPolynomial, scale: f32, illuminant: &'static dyn Spectrum) -> Self {
        Self {
            spd: rep,
            scale,
            illuminant,
        }
    }
    pub fn sample(&self, swl: SampledWavelengths) -> SampledSpectrum {
        self.spd.sample(swl) * self.scale * self.illuminant.sample(swl)
    }
}
impl DenselySampledSpectrum2 {
    pub const CIE_X: DenselySampledSpectrum2 = DenselySampledSpectrum2 {
        f: DenseSlice1D::new(
            (akari_const::CIE_LAMBDA_MIN, akari_const::CIE_LAMBDA_MAX),
            &akari_const::CIE_X,
        ),
    };
    pub const CIE_Y: DenselySampledSpectrum2 = DenselySampledSpectrum2 {
        f: DenseSlice1D::new(
            (akari_const::CIE_LAMBDA_MIN, akari_const::CIE_LAMBDA_MAX),
            &akari_const::CIE_Y,
        ),
    };
    pub const CIE_Z: DenselySampledSpectrum2 = DenselySampledSpectrum2 {
        f: DenseSlice1D::new(
            (akari_const::CIE_LAMBDA_MIN, akari_const::CIE_LAMBDA_MAX),
            &akari_const::CIE_Z,
        ),
    };
}

#[derive(Clone)]
pub struct GenericSpectrum<F: Function1D> {
    f: F,
}
impl<F> From<F> for GenericSpectrum<F>
where
    F: Function1D + 'static,
{
    fn from(f: F) -> Self {
        Self { f }
    }
}
impl<F: Function1D + 'static> Spectrum for GenericSpectrum<F> {
    fn sample(&self, swl: SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::new(vec4(
            self.f.evaluate(swl[0]),
            self.f.evaluate(swl[1]),
            self.f.evaluate(swl[2]),
            self.f.evaluate(swl[3]),
        ))
    }
}
pub type PiecewiseLinearSpectrum = GenericSpectrum<PiecewiseLinear1D>;
pub type DenselySampledSpectrum = GenericSpectrum<Dense1D>;
pub type DenselySampledSpectrum2 = GenericSpectrum<DenseSlice1D>;
impl PiecewiseLinearSpectrum {
    pub fn from_interleaved(xy: &[f32], normalize: bool) -> Self {
        let mut s: Self = PiecewiseLinear1D::from_interleaved(xy).into();
        if !normalize {
            return s;
        }
        let y = &DenselySampledSpectrum2::CIE_Y;
        s.scale(CIE_Y_INTEGRAL / inner_product(&s, &y));
        s
    }
    pub fn scale(&mut self, k: f32) {
        self.f.scale(k);
    }
}
pub trait Spectrum: Base + Send + Sync {
    fn sample(&self, lambda: SampledWavelengths) -> SampledSpectrum;
}
pub fn inner_product<F1: Function1D, F2: Function1D>(
    s1: &GenericSpectrum<F1>,
    s2: &GenericSpectrum<F2>,
) -> f32 {
    function::inner_product::<F1, F2>(&s1.f, &s2.f, 1.0)
}
fn init_spectrum_data() -> HashMap<&'static str, Arc<dyn Spectrum>> {
    log::info!("initialize spectrum data");
    let mut map: HashMap<&'static str, Arc<dyn Spectrum>> = HashMap::new();
    let illuma = PiecewiseLinearSpectrum::from_interleaved(&akari_const::CIE_ILLUM_A, true);
    let illumd65 = PiecewiseLinearSpectrum::from_interleaved(&akari_const::CIE_ILLUM_D65, true);
    map.insert("stdillum-A", Arc::new(illuma));
    map.insert("stdillum-D65", Arc::new(illumd65));
    map
}
lazy_static! {
    static ref NAMED_SPECTRUM: HashMap<&'static str, Arc<dyn Spectrum>> = init_spectrum_data();
}
pub fn spectrum_from_name(name: &str) -> &'static dyn Spectrum {
    NAMED_SPECTRUM
        .get(name)
        .expect(&format!("invalid spectrum {}", name))
        .as_ref()
}
