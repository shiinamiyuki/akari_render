use crate::{
    geometry::{Frame, ShadingTriangle},
    svm::ShaderRef,
    *,
};

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct SurfacePoint {
    pub p: Float3,
    pub n: Float3,
}

#[derive(Clone, Copy, Debug, Soa, Value)]
#[repr(C)]
pub struct SurfaceInteraction {
    pub frame: Frame,
    pub p: Float3,
    pub ng: Float3,
    pub bary: Float2,
    pub uv: Float2,
    pub inst_id: u32,
    pub prim_id: u32,
    pub surface: ShaderRef,
    pub valid: bool,
}
impl SurfaceInteractionExpr {
    pub fn ns(&self) -> Expr<Float3> {
        self.frame.n
    }
}
