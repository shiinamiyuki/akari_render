pub use luisa::derive::{Aggregate, Value};
pub use luisa::glam;
pub use luisa::prelude::*;
pub use luisa::resource::Sampler as TextureSampler;
pub use luisa::{
    lang::*,
    macros::*,
    math::*,
    resource::{BindlessArray, Buffer, BufferView, Tex2d, Tex2dView, Tex3d, Tex3dView},
    rtx,
    runtime::{Command, CommandBuffer, Device, Kernel, Stream, SyncHandle},
};
pub use luisa_compute as luisa;
pub mod camera;
pub mod color;
pub mod film;
pub mod filter;
pub mod geometry;
pub mod integrator;
pub mod interaction;
pub mod light;
pub mod sampler;
pub mod sampling;
pub mod scene;
pub mod scenegraph;
pub mod surface;
pub mod texture;
pub mod util;
pub mod volume;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum PolyKey {
    Simple(String),
    Dag(String, Vec<PolyKey>),
}
