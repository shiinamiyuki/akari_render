use crate::{geometry::Frame, svm::ShaderRef, *};

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct SurfacePoint {
    pub p: Float3,
    pub n: Float3,
}

#[derive(Clone, Copy, Aggregate)]
#[repr(C)]
pub struct SurfaceInteraction {
    pub frame: Expr<Frame>,
    pub p: Expr<Float3>,
    pub ng: Expr<Float3>,
    pub bary: Expr<Float2>,
    pub uv: Expr<Float2>,
    pub inst_id: Expr<u32>,
    pub prim_id: Expr<u32>,
    pub surface: Expr<ShaderRef>,
    pub prim_area: Expr<f32>,
    pub valid: Expr<bool>,
}
impl SurfaceInteraction {
    pub fn ns(&self) -> Expr<Float3> {
        self.frame.n
    }
    pub fn invalid() -> Self {
        Self {
            frame: Expr::<Frame>::zeroed(),
            p: Expr::<Float3>::zeroed(),
            ng: Expr::<Float3>::zeroed(),
            bary: Expr::<Float2>::zeroed(),
            uv: Expr::<Float2>::zeroed(),
            inst_id: Expr::<u32>::zeroed(),
            prim_id: Expr::<u32>::zeroed(),
            surface: Expr::<ShaderRef>::zeroed(),
            prim_area: Expr::<f32>::zeroed(),
            valid: false.expr(),
        }
    }
}
