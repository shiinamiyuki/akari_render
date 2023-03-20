use crate::*;

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct Ray {
    pub o: Float3,
    pub d: Float3,
    pub t_min: f32,
    pub t_max: f32,
}

impl RayExpr {
    pub fn at(&self, t: Float) -> Expr<Float3> {
        self.o() + self.d() * t
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct Triangle {
    pub v0: Float3,
    pub v1: Float3,
    pub v2: Float3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct ShadingTriangle {
    pub v0: Float3,
    pub v1: Float3,
    pub v2: Float3,
    pub tc0: Float2,
    pub tc1: Float2,
    pub tc2: Float2,
    pub ns0: Float3,
    pub ns1: Float3,
    pub ns2: Float3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct Frame {
    pub n: Float3,
    pub s: Float3,
    pub t: Float3,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct AffineTransform {
    pub m: Mat4,
    pub m_inv: Mat4,
    pub m3: Mat3,
    pub m3_inv: Mat3,
}
impl AffineTransformExpr {
    pub fn transform_point(&self, p: Expr<Float3>) -> Expr<Float3> {
        let q = make_float4(p.x(), p.y(), p.z(), 1.0);
        let q = self.m() * q;
        q.xyz() / q.w()
    }
    pub fn transform_vector(&self, v: Expr<Float3>) -> Expr<Float3> {
        self.m3() * v
    }
    pub fn transform_normal(&self, n: Expr<Float3>) -> Expr<Float3> {
        self.m3_inv().transpose() * n
    }
    pub fn inverse(&self) -> Self {
        Self::new(self.m_inv(), self.m(), self.m3_inv(), self.m3())
    }
}
