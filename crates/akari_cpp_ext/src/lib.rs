#[allow(non_snake_case)]
pub mod binding;

pub(crate) use akari_common::rayon;
pub use binding::root::{
    RayonScope, MLoopTri, Mesh, blender_util, spectral,
};
use std::ffi::c_void;

impl RayonScope {
    pub fn new<'a>(s: &rayon::Scope<'a>) -> Self {
        unsafe extern "C" fn spawn<'a>(
            context: *const c_void,
            func: Option<unsafe extern "C" fn(context:*const c_void, data: *mut c_void)>,
            data: *mut c_void,
        ) {
            let s = context as *const rayon::Scope<'a>;
            let s = &*s;
            let data = data as u64;
            s.spawn(move |s| {
                func.unwrap()(s as *const _ as *const c_void, data as *mut c_void);
            });
        }
        Self {
            num_threads: rayon::current_num_threads(),
            context: s as *const rayon::Scope<'a> as *const c_void,
            _spawn: Some(spawn),
        }
    }
}

unsafe impl Send for RayonScope {}

unsafe impl Sync for RayonScope {}
