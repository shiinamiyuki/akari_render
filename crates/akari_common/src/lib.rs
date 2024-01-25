pub use base64;
pub use bevy_mikktspace as mikktspace;
pub use clap;
pub use exr;
pub use glam;
pub use hexf;
pub use image;
pub use indicatif;
pub use lazy_static;
pub use libc;
pub use libm;
pub use log;
pub use luisa_compute as luisa;
pub use memmap2;
pub use parking_lot;
pub use rand;
pub use rayon;
pub use serde;
pub use serde_json;
pub use sha2;
pub use winit;
pub use ddsfile;

pub fn catch_signal(f: impl FnOnce()) {
    fn sigsegv_handler(_: libc::c_int) {
        std::panic::panic_any("Received SIGSEGV");
    }
    unsafe {
        let old = libc::signal(libc::SIGSEGV, sigsegv_handler as usize);
        f();
        libc::signal(libc::SIGSEGV, old as usize);
    }
}
