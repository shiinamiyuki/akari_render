use crate::*;
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct PointNormal {
    pub p: Float3,
    pub n: Float3,
}
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
    pub ng: Float3,
}

impl ShadingTriangleExpr {
    pub fn p(&self, bary: Expr<Float2>) -> Expr<Float3> {
        (1.0 - bary.x() - bary.y()) * self.v0() + bary.x() * self.v1() + bary.y() * self.v2()
    }
    pub fn n(&self, bary: Expr<Float2>) -> Expr<Float3> {
        ((1.0 - bary.x() - bary.y()) * self.ns0() + bary.x() * self.ns1() + bary.y() * self.ns2())
            .normalize()
    }
    pub fn tc(&self, bary: Expr<Float2>) -> Expr<Float2> {
        (1.0 - bary.x() - bary.y()) * self.tc0() + bary.x() * self.tc1() + bary.y() * self.tc2()
    }
    pub fn area(&self) -> Float {
        0.5 * (self.v1() - self.v0())
            .cross(self.v2() - self.v0())
            .length()
    }
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
    pub fn sin_phi(w: Expr<Float3>) -> Float {
        let sin_theta = Self::sin_theta(w);
        select(
            sin_theta.cmpeq(0.0),
            const_(1.0f32),
            (w.x() / sin_theta).clamp(-1.0, 1.0),
        )
    }
    #[inline]
    pub fn sin2_phi(w: Expr<Float3>) -> Float {
        let sin_theta = Self::sin_theta(w);
        select(
            sin_theta.cmpeq(0.0),
            const_(0.0f32),
            (w.x() / sin_theta).clamp(-1.0, 1.0).sqr(),
        )
    }
    #[inline]
    pub fn cos2_phi(w: Expr<Float3>) -> Float {
        let sin_theta = Self::sin_theta(w);
        select(
            sin_theta.cmpeq(0.0),
            const_(0.0f32),
            (w.z() / sin_theta).clamp(-1.0, 1.0).sqr(),
        )
    }
    #[inline]
    pub fn cos_phi(w: Expr<Float3>) -> Float {
        let sin_theta = Self::sin_theta(w);
        select(
            sin_theta.cmpeq(0.0),
            const_(1.0f32),
            (w.z() / sin_theta).clamp(-1.0, 1.0),
        )
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
impl AffineTransform {
    pub fn from_matrix(m: &glam::Mat4) -> Self {
        let m3 = glam::Mat3::from_mat4(*m);
        Self {
            m: (*m).into(),
            m_inv: m.inverse().into(),
            m3: m3.into(),
            m3_inv: m3.inverse().into(),
        }
    }
    pub fn inverse(&self) -> Self {
        Self {
            m: self.m_inv,
            m_inv: self.m,
            m3: self.m3_inv,
            m3_inv: self.m3,
        }
    }
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
pub fn face_forward(v: Expr<Float3>, n: Expr<Float3>) -> Expr<Float3> {
    select(v.dot(n).cmplt(0.0), -v, v)
}
pub fn reflect(w: Expr<Float3>, n: Expr<Float3>) -> Expr<Float3> {
    -w + 2.0 * w.dot(n)
}
pub fn refract(w: Expr<Float3>, mut n: Expr<Float3>, eta: Expr<f32>) -> (Expr<bool>, Expr<Float3>) {
    let mut cos_theta_i = w.dot(n);
    let entering = cos_theta_i.cmpgt(0.0);
    let (eta, cos_theta_i, n) = if_!(entering, {
        (eta, cos_theta_i, n)
    }, else {
        let eta = 1.0 / eta;
        let cos_theta_i = -cos_theta_i;
        let n = -n;
        (eta, cos_theta_i, n)
    });
    let sin2_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0);
    let sin2_theta_t = sin2_theta_i / (eta * eta);
    let refracted = sin2_theta_t.cmplt(1.0);
    if_!(refracted, {
        (const_(false), Float3Expr::zero())
    }, else {
        todo!()
    })
}
