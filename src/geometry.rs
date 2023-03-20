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

impl Frame {
    #[inline]
    pub fn cos_theta(w: Expr<Float3>) -> Float {
        w.y()
    }
    #[inline]
    pub fn cos2_theta(w: Expr<Float3>) -> Float {
        let c = w.y();
        c * c
    }
    #[inline]
    pub fn abs_cos_theta(w: Expr<Float3>) -> Float {
        Self::cos_theta(w).abs()
    }
    #[inline]
    pub fn sin_theta(w: Expr<Float3>) -> Float {
        let c = Self::cos_theta(w);
        (1.0 - c * c).max(0.0).sqrt()
    }
    #[inline]
    pub fn sin2_theta(w: Expr<Float3>) -> Float {
        let c = Self::cos_theta(w);
        (1.0 - c * c).max(0.0)
    }
    #[inline]
    pub fn tan2_theta(w: Expr<Float3>) -> Float {
        Self::sin2_theta(w) / Self::cos2_theta(w)
    }
    #[inline]
    pub fn tan_theta(w: Expr<Float3>) -> Float {
        Self::sin_theta(w) / Self::cos_theta(w)
    }
    #[inline]
    pub fn same_hemisphere(w1: Expr<Float3>, w2: Expr<Float3>) -> Bool {
        (w1.y() * w2.y()).cmpge(0.0)
    }
}
impl FrameExpr {
    pub fn from_n(n: Expr<Float3>) -> Self {
        let t = if_!(n.x().abs().cmpgt(n.y().abs()), {
            make_float3(-n.z(), 0.0, n.x()) / (n.x() * n.x() + n.z() * n.z()).sqrt()
        }, else {
            make_float3(0.0, n.z(), -n.y()) / (n.y() * n.y() + n.z() * n.z()).sqrt()
        });
        let s = n.cross(t);
        Self::new(n, s, t)
    }
    pub fn to_world(&self, v: Expr<Float3>) -> Expr<Float3> {
        self.s() * v.x() + self.n() * v.y() + self.t() * v.z()
    }
    pub fn to_local(&self, v: Expr<Float3>) -> Expr<Float3> {
        make_float3(self.s().dot(v), self.n().dot(v), self.t().dot(v))
    }
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
