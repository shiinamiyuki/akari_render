#[allow(non_snake_case)]
pub mod binding;

use crate::binding::root::ParallelForFn;
pub(crate) use akari_common::rayon;
pub use binding::root::{blender_util, spectral, MLoopTri, Mesh, ParallelForContext};
use std::ffi::c_void;
use std::sync::atomic::AtomicUsize;

impl ParallelForContext {
    pub fn new() -> Self {
        unsafe extern "C" fn parallel_for_impl(
            count: usize,
            f: ParallelForFn,
            userdata: *const c_void,
        ) {
            let userdata = userdata as u64;
            let num_threads = rayon::current_num_threads();
            let cnt = AtomicUsize::new(0);
            let f = f.unwrap();
            rayon::scope(|s| {
                for _ in 0..num_threads {
                    let cnt = &cnt;
                    s.spawn(move |_| {
                        let userdata = userdata as *const c_void;
                        loop {
                            let i = cnt.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= count {
                                break;
                            }
                            f(userdata, i);
                        }
                    });
                }
            });
        }
        Self {
            _parallel_for: Some(parallel_for_impl),
        }
    }
}
