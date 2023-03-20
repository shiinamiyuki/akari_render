use crate::{*, geometry::{ShadingTriangle, Frame}, scene::Scene, color::ColorRepr};
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
    pub uv: Float2,
    pub dpdu: Float3,
    pub dpdv: Float3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfaceInteraction {
    pub geometry: SurfaceLocalGeometry,
    pub uv: Float2,
    pub prim_id: u32,
    pub inst_id: u32,
    pub frame: Frame,
    pub triangle: ShadingTriangle,
}


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    LightToEye,
    EyeToLight,
}

pub struct ShadingContext<'a> {
    pub scene: &'a Scene,
    pub color_repr: ColorRepr,
}