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
    runtime::{create_static_callable, Callable,DynCallable, Command, Device, Kernel, Stream},
};
pub use luisa_compute as luisa;
pub use rayon::prelude::*;
pub mod camera;
pub mod color;
pub mod film;
pub mod filter;
pub mod geometry;
pub mod integrator;
pub mod interaction;
pub mod light;
pub mod mesh;
mod microfacet;
pub mod sampler;
pub mod sampling;
pub mod scene;
pub mod scenegraph;
pub mod surface;
pub mod texture;
pub mod util;
pub mod volume;
pub mod ext;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum PolyKey {
    Simple(String),
    Dag(String, Vec<PolyKey>),
}
