#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
pub mod bindings;
pub use bindings::*;
use lazy_static::lazy_static;
use akari_image_PixelFormat as PixelFormat;
fn load_library() -> AkariCppExt {
    const LIB_ENV: &str = "AKARI_CPP_EXT_LIB";
    const LIB_NAME: &str = "akari_cpp_ext.dll";
    // get the library path from the environment variable
    match std::env::var(LIB_ENV) {
        Ok(lib_path) => {
            return unsafe { AkariCppExt::new(lib_path).unwrap() };
        }
        Err(_) => {
            let exe_path = std::env::current_exe().unwrap();
            let exe_dir = exe_path.parent().unwrap();
            let lib_path = exe_dir.join(LIB_NAME);
            return unsafe {
                AkariCppExt::new(lib_path.clone()).unwrap_or_else(|e| {
                    eprintln!("Failed to load {}: {}", lib_path.display(), e);
                    std::process::exit(1);
                })
            };
        }
    }
}

lazy_static! {
    static ref CPP_EXT: AkariCppExt = load_library();
}
pub fn extension() -> &'static AkariCppExt {
    &*CPP_EXT
}

impl PixelFormat {
    #[inline]
    pub fn size(self) -> usize {
        match self {
            PixelFormat::R8 => 1,
            PixelFormat::RF32 => 4,
            PixelFormat::RGBA8 => 4,
            PixelFormat::RGBF32 => 12,
            PixelFormat::RGBAF32 => 16,
            _ => panic!("Invalid pixel format {:?}", self),
        }
    }
}
