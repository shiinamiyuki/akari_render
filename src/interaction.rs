use crate::{*, geometry::{ShadingTriangle, Frame}, scene::Scene, color::ColorRepr};
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfacePoint {
    pub p: Vec3,
    pub n: Vec3,
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfaceLocalGeometry {
    pub p: Vec3,
    pub ng: Vec3,
    pub ns: Vec3,
    pub uv: Vec2,
    pub dpdu: Vec3,
    pub dpdv: Vec3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SurfaceInteraction {
    pub geometry: SurfaceLocalGeometry,
    pub uv: Vec2,
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