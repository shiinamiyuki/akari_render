use crate::*;
pub const SPECTRUM_TABLE_RES: usize = 64;
pub type Rgb2SpectrumTable =
    [[[[[f32; 3]; SPECTRUM_TABLE_RES]; SPECTRUM_TABLE_RES]; SPECTRUM_TABLE_RES]; 3];
pub type Rgb2SpectrumScale = [f32; SPECTRUM_TABLE_RES];

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
pub fn load_rgb2spec_data(gamut: &str, file: String) -> Arc<Rgb2SpectrumData> {
    let exe_path = std::env::current_exe().unwrap();
    let parent = exe_path.parent().unwrap();
    let mut data_file = PathBuf::from(parent);
    data_file.push(&file);
    let data_file = data_file.into_boxed_path();

    if !data_file.exists() {
        {
            log::info!("{} does not exist, generating...", data_file.display());
            // let mut scale = Box::new([0.0f32; SPECTRUM_TABLE_RES]);
            // let mut table = Box::new(
            //     [[[[[0.0f32; 3]; SPECTRUM_TABLE_RES]; SPECTRUM_TABLE_RES]; SPECTRUM_TABLE_RES]; 3],
            // );
            // let gamut = CString::new(gamut).unwrap();
            // gen_rgb2spec_table(
            //     SPECTRUM_TABLE_RES as u32,
            //     gamut.as_ptr(),
            //     scale.as_mut_ptr(),
            //     table.as_mut_ptr() as *mut f32,
            // );
            // let mut file = std::fs::File::create(&data_file).unwrap();
            // file.write_all(std::slice::from_raw_parts(
            //     scale.as_ptr() as *const u8,
            //     size_of::<Rgb2SpectrumScale>(),
            // ))
            // .unwrap();
            // file.write_all(std::slice::from_raw_parts(
            //     table.as_ptr() as *const u8,
            //     size_of::<Rgb2SpectrumTable>(),
            // ))
            // .unwrap();
            let path = current_exe().unwrap();
            let mut path = PathBuf::from(path.parent().unwrap());
            path.push("rgb2spec_opt");
            // dbg!(path.display());
            let data_file = data_file.as_os_str().to_str().unwrap();
            if !Command::new(path)
                .args(["64", data_file, gamut])
                .spawn()
                .expect("failed to launch rgb2spec_opt")
                .wait()
                .unwrap()
                .success()
            {
                eprintln!("failed to generate rgb2spec data");
                exit(-1);
            }
        }
        load_rgb2spec_data(gamut, file)
    } else {
        log::info!("loading spectrum from {}", data_file.display());
        let mut file = std::fs::File::open(&data_file).unwrap();
        let mut data = Arc::new(Rgb2SpectrumData::_DUMMY);
        {
            let data = Arc::get_mut(&mut data).unwrap();
            unsafe {
                file.read_exact(std::slice::from_raw_parts_mut(
                    data.scale.as_mut_ptr() as *mut u8,
                    size_of::<Rgb2SpectrumScale>(),
                ))
                .unwrap();
                file.read_exact(std::slice::from_raw_parts_mut(
                    data.table.as_mut_ptr() as *mut u8,
                    size_of::<Rgb2SpectrumTable>(),
                ))
                .unwrap();
            }
        }
        data
    }
}

use std::{
    env::current_exe,
    ffi::CString,
    io::{Read, Write},
    mem::size_of,
    os::raw::c_char,
    path::PathBuf,
    process::{exit, Command},
    sync::Arc,
};

macro_rules! rgb2spec_mod {
    ($name:ident) => {
        pub mod $name {
            use super::*;
            use lazy_static::lazy_static;
            lazy_static! {
                pub static ref DATA: Arc<Rgb2SpectrumData> = load_rgb2spec_data(
                    stringify!($name),
                    format!("rgbspectrum_{}", stringify!($name))
                );
            }
        }
    };
}

rgb2spec_mod!(aces2065_1);
rgb2spec_mod!(dci_p3);
rgb2spec_mod!(srgb);
rgb2spec_mod!(rec2020);
