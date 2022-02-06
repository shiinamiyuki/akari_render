use crate::*;
pub const SPECTRUM_TABLE_RES: usize = 64;
pub type Rgb2SpectrumTable =
    [[[[[f32; 3]; SPECTRUM_TABLE_RES]; SPECTRUM_TABLE_RES]; SPECTRUM_TABLE_RES]; 3];
pub type Rgb2SpectrumScale = [f32; SPECTRUM_TABLE_RES];
use std::{
    ffi::CString,
    io::{Read, Write},
    mem::size_of,
    os::raw::c_char,
    path::PathBuf, sync::Arc,
};

pub struct Rgb2SpectrumData {
    pub scale: Rgb2SpectrumScale,
    pub table: Rgb2SpectrumTable,
}
impl Rgb2SpectrumData {
    pub const _DUMMY: Self = Self {
        scale: [0.0f32; SPECTRUM_TABLE_RES],
        table: [[[[[0.0f32; 3]; SPECTRUM_TABLE_RES]; SPECTRUM_TABLE_RES]; SPECTRUM_TABLE_RES]; 3],
    };
}

fn load_rgb2spec_data(bytes: &[u8]) -> Arc<Rgb2SpectrumData> {
    let raw = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / size_of::<f32>())
    };
    assert_eq!(bytes.len(), size_of::<Rgb2SpectrumData>());
    let mut data = Arc::new(Rgb2SpectrumData::_DUMMY);
    unsafe {
        let data = Arc::get_mut(&mut data).unwrap();
        std::ptr::copy_nonoverlapping(raw.as_ptr(), data.scale.as_mut_ptr(), SPECTRUM_TABLE_RES);
        std::ptr::copy_nonoverlapping(
            raw[SPECTRUM_TABLE_RES..].as_ptr(),
            data.table.as_mut_ptr() as *mut f32,
            SPECTRUM_TABLE_RES.pow(3) * 9,
        );
    }
    data
}

macro_rules! rgb2spec_mod {
    ($name:ident, $file:literal) => {
        pub mod $name {
            use super::*;
            use lazy_static::lazy_static;
            lazy_static! {
                pub static ref DATA: Arc<Rgb2SpectrumData> =
                    load_rgb2spec_data(include_bytes!($file));
            }
        }
    };
}

rgb2spec_mod!(aces2065_1, "rgbspectrum_aces2065_1");
rgb2spec_mod!(dci_p3, "rgbspectrum_dci_p3");
rgb2spec_mod!(srgb, "rgbspectrum_srgb");
rgb2spec_mod!(rec2020, "rgbspectrum_rec2020");
