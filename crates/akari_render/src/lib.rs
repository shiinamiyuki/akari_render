use std::env;

pub use akari_common::luisa;
use color::ColorRepr;

pub use akari_common::*;
pub use akari_scenegraph as scene_graph;
pub(crate) use hexf::hexf32;
pub(crate) use lazy_static::lazy_static;
pub use luisa::prelude::{
    device_log as _device_log, lc_assert as _lc_assert, outline, track as _track, tracked,
    Aggregate, Soa, Value,
};
pub use luisa::resource::Sampler as TextureSampler;
pub use luisa::{
    impl_polymorphic as _impl_polymorphic,
    lang::{
        control_flow::*,
        functions::*,
        index::*,
        ops::*,
        poly::*,
        soa::*,
        types::{array::*, core::*, vector::alias::*, vector::swizzle::*, vector::*, *},
    },
    resource::{
        BindlessArray, BindlessArrayVar, BindlessBufferVar, BindlessTex2dVar, BindlessTex3dVar,
        Buffer, BufferVar, BufferView, ByteBuffer, ByteBufferVar, ByteBufferView, PixelStorage,
        SamplerAddress, SamplerFilter, Tex2d, Tex2dVar, Tex2dView, Tex3d, Tex3dVar, Tex3dView,
    },
    rtx::{self, TriangleInterpolate},
    runtime::{
        api::StreamTag, Callable, Command, Device, DynCallable, Kernel, Scope, Stream, Swapchain,
    },
};
pub use rayon::prelude::*;
pub mod api;
pub mod camera;
pub mod color;
pub mod cpp_ext;
pub mod data;
pub mod ext;
pub mod film;
pub mod geometry;
pub mod gui;
pub mod heap;
pub mod interaction;
pub mod light;
pub mod load;
pub mod mesh;
pub mod microfacet;
pub mod sampler;
pub mod sampling;
pub mod scene;
pub mod svm;
pub mod util;
pub mod volume;

pub const ONE_MINUS_EPSILON: f32 = hexf32!("0x1.fffffep-1");
pub const FRAC_1_2PI: f32 = 1.0 / (2.0 * std::f32::consts::PI);

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum ADMode {
    None,
    Forward,
    Backward,
}
pub fn maybe_outline(f: impl Fn()) {
    let should_not_outline = match env::var("AKR_NO_OUTLINE") {
        Ok(x) => x == "1",
        _ => false,
    };
    if !should_not_outline {
        outline(f);
    } else {
        f();
    }
}

pub fn debug_mode() -> bool {
    cfg!(debug_assertions)
        || match env::var("LUISA_DEBUG") {
            Ok(x) => {
                if x == "1" || x == "full" {
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
}
#[macro_export]
macro_rules! impl_polymorphic {
    ($trait_:ident, $ty:ty) => {
        _impl_polymorphic!(crate = [luisa], $trait_, $ty);
    };
}

#[macro_export]
macro_rules! lc_assert {
    ($arg:expr) => {
        _lc_assert!(crate = [luisa], $arg)
    };
    ($arg:expr, $msg:expr) => {
        _lc_assert!(crate = [luisa], $arg, $msg)
    };
}
#[macro_export]
macro_rules! device_log {
    ($fmt:literal, $($arg:expr),*) => {{
        _device_log!(crate=[luisa], $fmt, $($arg),*)
    }};
}
#[macro_export]
macro_rules! track {
    ($arg:expr) => {
        _track!(crate="luisa"=> $arg)
    };
}
