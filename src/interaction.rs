use crate::{
    geometry::{Frame, ShadingTriangle},
    *,
};

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfacePoint {
    pub p: Float3,
    pub n: Float3,
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfaceLocalGeometry {
    pub p: Float3,
    pub ng: Float3,
    pub ns: Float3,
    // uv in UV mapping
    pub uv: Float2,
    pub dpdu: Float3,
    pub dpdv: Float3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfaceInteraction {
    pub geometry: SurfaceLocalGeometry,
    pub bary: Float2,
    pub prim_id: u32,
    pub inst_id: u32,
    pub frame: Frame,
    pub triangle: ShadingTriangle,
    pub valid: bool,
}