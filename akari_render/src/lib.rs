use color::ColorRepr;
use hexf::hexf32;
use integrator::FilmConfig;
pub use luisa::derive::{Aggregate, Value};
pub use luisa::glam;
pub use luisa::prelude::*;
pub use luisa::resource::Sampler as TextureSampler;
pub use luisa::{
    lang::*,
    macros::*,
    math::*,
    resource::{
        BindlessArray, BindlessArrayVar, BindlessBufferVar, BindlessTex2dVar, BindlessTex3dVar,
        Buffer, BufferVar, BufferView, ByteBuffer, ByteBufferVar, ByteBufferView, Tex2d, Tex2dVar,
        Tex2dView, Tex3d, Tex3dVar, Tex3dView,
    },
    rtx,
    runtime::{create_static_callable, Callable, Command, Device, DynCallable, Kernel, Stream},
};
pub use luisa_compute as luisa;
pub use rayon::prelude::*;
pub mod camera;
pub mod color;
pub mod cpp_ext;
pub mod data;
pub mod ext;
pub mod film;
pub mod filter;
pub mod geometry;
pub mod integrator;
pub mod interaction;
pub mod light;
pub mod load;
pub mod mesh;
pub mod microfacet;
pub mod nodes;
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
    Forward,
    Backward,
}
