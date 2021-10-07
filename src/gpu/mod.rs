pub mod accel;
pub mod mesh;
pub mod pt;
pub mod scene;
pub mod soa;

#[cfg(feature = "gpu_nrc")]
pub mod nrc;
#[cfg(feature = "gpu_nrc")]
pub mod nrc_sys;
use std::{
    borrow::Cow,
    ffi::{c_void, CStr, CString},
    os::raw::c_char,
    process::abort,
    sync::Arc,
};
