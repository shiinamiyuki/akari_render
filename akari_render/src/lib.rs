use hexf::hexf32;
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
        Buffer, BufferVar, BufferView, Tex2d, Tex2dVar, Tex2dView, Tex3d, Tex3dVar, Tex3dView,
    },
    rtx,
    runtime::{create_static_callable, Callable, Command, Device, DynCallable, Kernel, Stream},
};
pub use luisa_compute as luisa;
pub use rayon::prelude::*;
pub mod camera;
pub mod color;
pub mod ext;
pub mod film;
pub mod filter;
pub mod geometry;
pub mod integrator;
pub mod interaction;
pub mod light;
pub mod mesh;
pub mod microfacet;
pub mod sampler;
pub mod sampling;
pub mod scene;
pub mod scenegraph;
pub mod surface;
pub mod texture;
pub mod util;
pub mod data;
pub mod cpp_ext;
pub mod volume;
pub mod nodes;

pub const ONE_MINUS_EPSILON: f32 = hexf32!("0x1.fffffep-1");
pub const FRAC_1_2PI: f32 = 1.0 / (2.0 * std::f32::consts::PI);

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum PolyKey {
    Simple(String),
    Dag(String, Vec<PolyKey>),
}
