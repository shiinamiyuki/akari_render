use crate::{
    geometry::{Frame, ShadingTriangle},
    *,
};

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct SurfacePoint {
    pub p: Float3,
    pub n: Float3,
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct SurfaceLocalGeometry {
    pub p: Float3,
    pub ng: Float3,
    pub ns: Float3,
    pub tangent: Float3,
    pub bitangent: Float3,
    pub uv: Float2,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct SurfaceInteraction {
    pub inst_id: u32,
    pub prim_id: u32,
    pub bary: Float2,
    pub geometry: SurfaceLocalGeometry,
    pub frame: Frame,
    pub valid: bool,
}
