use std::ffi::c_char;

use akari_render::api;
use std::ffi::CString;

#[no_mangle]
pub unsafe extern "cdecl" fn py_akari_import(api: *const c_char, len: u64) -> *mut c_char {
    let api_json = std::slice::from_raw_parts(api as *const u8, len as usize);
    let api_json = std::str::from_utf8(api_json).unwrap();
    let api: api::SceneImportApi = serde_json::from_str(api_json).unwrap_or_else(|err| {
        eprintln!("Failed to parse JSON: {}\n JSON: `{}`", err, api_json);
        std::process::exit(1);
    });
    let result = api::import(api);
    let result_json = serde_json::to_string(&result).unwrap();
    let result_json = std::ffi::CString::new(result_json).unwrap();
    let ptr = result_json.into_raw();
    ptr
}

#[no_mangle]
pub unsafe extern "cdecl" fn py_akari_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}
