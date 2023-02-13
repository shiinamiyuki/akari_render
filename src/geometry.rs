use crate::*;

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,
    pub t_min: f32,
    pub t_max: f32,
}

impl RayExpr {
    pub fn at(&self, t: Float32) -> Expr<Vec3> {
        self.o() + self.d() * t
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct ShadingTriangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub tc0: Vec2,
    pub tc1: Vec2,
    pub tc2: Vec2,
    pub ns0: Vec3,
    pub ns1: Vec3,
    pub ns2: Vec3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct Frame {
    pub n: Vec3,
    pub s: Vec3,
    pub t: Vec3,
}