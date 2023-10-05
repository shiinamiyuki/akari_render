use std::env;

use color::ColorRepr;
use hexf::hexf32;
use integrator::FilmConfig;
pub use luisa::prelude::{
    cpu_dbg, if_, lc_assert, lc_comment_lineno, lc_unreachable, loop_, track, tracked, while_,
    Aggregate, Value,
};
pub use luisa::resource::Sampler as TextureSampler;
pub use luisa::{
    impl_polymorphic,
    lang::{
        control_flow::*,
        functions::*,
        index::*,
        ops::*,
        poly::*,
        types::{array::*, core::*, vector::alias::*, vector::swizzle::*, vector::*, *},
    },
    lc_info,
    printer::Printer,
    resource::{
        BindlessArray, BindlessArrayVar, BindlessBufferVar, BindlessTex2dVar, BindlessTex3dVar,
        Buffer, BufferVar, BufferView, ByteBuffer, ByteBufferVar, ByteBufferView, PixelStorage,
        SamplerAddress, SamplerFilter, Tex2d, Tex2dVar, Tex2dView, Tex3d, Tex3dVar, Tex3dView,
    },
    rtx,
    runtime::{api::StreamTag, Callable, Command, Device, DynCallable, Kernel, Stream, Swapchain},
};
pub use luisa_compute as luisa;
pub use rayon::prelude::*;
pub mod camera;
pub mod color;
pub mod cpp_ext;
pub mod data;
pub mod ext;
pub mod film;
pub mod geometry;
pub mod gui;
pub mod integrator;
pub mod interaction;
pub mod light;
pub mod load;
pub mod mesh;
pub mod microfacet;
pub mod node;
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
