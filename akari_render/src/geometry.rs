use std::f32::consts::PI;

use crate::*;
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct PointNormal {
    pub p: Float3,
    pub n: Float3,
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct Ray {
    pub o: Float3,
    pub d: Float3,
    pub t_min: f32,
    pub t_max: f32,
    pub exclude0: Uint2,
    pub exclude1: Uint2,
}

impl RayExpr {
    #[tracked]
    pub fn at(&self, t: Expr<f32>) -> Expr<Float3> {
        self.o + self.d * t
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct Sphere {
    pub center: Float3,
    pub radius: f32,
}
impl SphereExpr {
    #[tracked]
    pub fn intersect(&self, ray: Expr<Ray>) -> (Expr<bool>, Expr<f32>) {
        let a = ray.d.length_squared();
        let oc = ray.o - self.center;
        let b = 2.0 * oc.dot(ray.d);
        let c = oc.length_squared() - self.radius.sqr();
        let delta = b.sqr() - 4.0 * a * c;
        if delta.lt(0.0) {
            (false.expr(), 0.0f32.expr())
        } else {
            let t0 = (-b - delta.sqrt()) / (2.0 * a);
            let t1 = (-b + delta.sqrt()) / (2.0 * a);
            if t0.lt(ray.t_max) & t0.ge(ray.t_min) {
                (true.expr(), t0)
            } else {
                if !t1.lt(ray.t_max) & t1.ge(ray.t_min) {
                    (true.expr(), t1)
                } else {
                    (false.expr(), 0.0f32.expr())
                }
            }
        }
    }
}
#[derive(Clone, Copy, Aggregate)]
#[repr(C)]
pub struct Triangle {
    pub v0: Expr<Float3>,
    pub v1: Expr<Float3>,
    pub v2: Expr<Float3>,
}

#[derive(Clone, Copy, Aggregate)]
#[repr(C)]
pub struct ShadingTriangle {
    pub v0: Expr<Float3>,
    pub v1: Expr<Float3>,
    pub v2: Expr<Float3>,
    pub uv0: Expr<Float2>,
    pub uv1: Expr<Float2>,
    pub uv2: Expr<Float2>,
    pub n0: Expr<Float3>,
    pub n1: Expr<Float3>,
    pub n2: Expr<Float3>,
    pub t0: Expr<Float3>,
    pub t1: Expr<Float3>,
    pub t2: Expr<Float3>,
    pub b0: Expr<Float3>,
    pub b1: Expr<Float3>,
    pub b2: Expr<Float3>,
    pub ng: Expr<Float3>,
}

impl ShadingTriangle {
    #[tracked]
    pub fn p(&self, bary: Expr<Float2>) -> Expr<Float3> {
        (1.0 - bary.x - bary.y) * self.v0 + bary.x * self.v1 + bary.y * self.v2
    }
    #[tracked]
    pub fn n(&self, bary: Expr<Float2>) -> Expr<Float3> {
        ((1.0 - bary.x - bary.y) * self.n0 + bary.x * self.n1 + bary.y * self.n2).normalize()
    }
    #[tracked]
    pub fn tangent(&self, bary: Expr<Float2>) -> Expr<Float3> {
        ((1.0 - bary.x - bary.y) * self.t0 + bary.x * self.t1 + bary.y * self.t2).normalize()
    }
    #[tracked]
    pub fn bitangent(&self, bary: Expr<Float2>) -> Expr<Float3> {
        ((1.0 - bary.x - bary.y) * self.b0 + bary.x * self.b1 + bary.y * self.b2).normalize()
    }
    #[tracked]
    pub fn uv(&self, bary: Expr<Float2>) -> Expr<Float2> {
        (1.0 - bary.x - bary.y) * self.uv0 + bary.x * self.uv1 + bary.y * self.uv2
    }
    #[tracked]
    pub fn area(&self) -> Expr<f32> {
        0.5 * (self.v1 - self.v0).cross(self.v2 - self.v0).length()
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct Frame {
    pub n: Float3,
    pub t: Float3,
    pub s: Float3,
}

impl Frame {
    #[inline]
    #[tracked]
    pub fn cos_theta(w: Expr<Float3>) -> Expr<f32> {
        w.y
    }
    #[inline]
    #[tracked]
    pub fn cos2_theta(w: Expr<Float3>) -> Expr<f32> {
        let c = w.y;
        c * c
    }
    #[inline]
    #[tracked]
    pub fn abs_cos_theta(w: Expr<Float3>) -> Expr<f32> {
        Self::cos_theta(w).abs()
    }
    #[inline]
    #[tracked]
    pub fn sin_theta(w: Expr<Float3>) -> Expr<f32> {
        let c2 = Self::cos2_theta(w);
        (1.0 - c2).max_(0.0).sqrt()
    }
    #[inline]
    #[tracked]
    pub fn sin2_theta(w: Expr<Float3>) -> Expr<f32> {
        let c2 = Self::cos2_theta(w);
        (1.0 - c2).max_(0.0)
    }
    #[inline]
    #[tracked]
    pub fn tan2_theta(w: Expr<Float3>) -> Expr<f32> {
        Self::sin2_theta(w) / Self::cos2_theta(w)
    }
    #[inline]
    #[tracked]
    pub fn tan_theta(w: Expr<Float3>) -> Expr<f32> {
        Self::sin_theta(w) / Self::cos_theta(w)
    }
    #[inline]
    #[tracked]
    pub fn sin_phi(w: Expr<Float3>) -> Expr<f32> {
        let sin_theta = Self::sin_theta(w);
        select(
            sin_theta.eq(0.0),
            0.0f32.expr(),
            (w.x / sin_theta).clamp(-1.0.expr(), 1.0.expr()),
        )
    }
    #[inline]
    #[tracked]
    pub fn sin2_phi(w: Expr<Float3>) -> Expr<f32> {
        Self::sin_phi(w).sqr()
    }
    #[inline]
    #[tracked]
    pub fn cos2_phi(w: Expr<Float3>) -> Expr<f32> {
        Self::cos_phi(w).sqr()
    }
    #[inline]
    #[tracked]
    pub fn cos_phi(w: Expr<Float3>) -> Expr<f32> {
        let sin_theta = Self::sin_theta(w);
        select(
            sin_theta.eq(0.0),
            1.0f32.expr(),
            (w.z / sin_theta).clamp(-1.0.expr(), 1.0.expr()),
        )
    }
    #[inline]
    #[tracked]
    pub fn same_hemisphere(w1: Expr<Float3>, w2: Expr<Float3>) -> Expr<bool> {
        (w1.y * w2.y) >= (0.0)
    }
}
impl FrameExpr {
    #[tracked]
    pub fn from_n(n: Expr<Float3>) -> Expr<Frame> {
        let t = if n.x.abs().gt(n.y.abs()) {
            Float3::expr(-n.z, 0.0, n.x) / (n.x * n.x + n.z * n.z).sqrt()
        } else {
            Float3::expr(0.0, n.z, -n.y) / (n.y * n.y + n.z * n.z).sqrt()
        };
        let s = n.cross(t);
        Frame::new_expr(n, s, t)
    }
    #[tracked]
    pub fn to_world(&self, v: Expr<Float3>) -> Expr<Float3> {
        self.s * v.x + self.n * v.y + self.t * v.z
    }
    #[tracked]
    pub fn to_local(&self, v: Expr<Float3>) -> Expr<Float3> {
        Float3::expr(self.s.dot(v), self.n.dot(v), self.t.dot(v))
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[value_new(pub)]
pub struct AffineTransform {
    pub m: Mat4,
    pub m_inv: Mat4,
    pub m3: Mat3,
    pub m3_inv: Mat3,
    pub close_to_identity: bool,
}
impl AffineTransform {
    pub fn from_matrix(m: &glam::Mat4) -> Self {
        let close_to_identity = m.abs_diff_eq(glam::Mat4::IDENTITY, 1e-4);
        let m3 = glam::Mat3::from_mat4(*m);
        Self {
            m: (*m).into(),
            m_inv: m.inverse().into(),
            m3: m3.into(),
            m3_inv: m3.inverse().into(),
            close_to_identity,
        }
    }
    pub fn inverse(&self) -> Self {
        Self {
            m: self.m_inv,
            m_inv: self.m,
            m3: self.m3_inv,
            m3_inv: self.m3,
            close_to_identity: self.close_to_identity,
        }
    }
}
impl AffineTransformExpr {
    #[tracked]
    pub fn transform_point(&self, p: impl AsExpr<Value = Float3>) -> Expr<Float3> {
        let p = p.as_expr();
        if self.close_to_identity {
            p
        } else {
            let q = Float4::expr(p.x, p.y, p.z, 1.0);
            let q = self.m * q;
            q.xyz() / q.w
        }
    }
    #[tracked]
    pub fn transform_vector(&self, v: impl AsExpr<Value = Float3>) -> Expr<Float3> {
        let v = v.as_expr();
        if self.close_to_identity {
            v
        } else {
            self.m3 * v
        }
    }
    #[tracked]
    pub fn transform_normal(&self, n: impl AsExpr<Value = Float3>) -> Expr<Float3> {
        let n = n.as_expr();
        if self.close_to_identity {
            n
        } else {
            self.m3_inv.transpose() * n
        }
    }
    pub fn inverse(&self) -> Expr<AffineTransform> {
        AffineTransform::new_expr(
            self.m_inv,
            self.m,
            self.m3_inv,
            self.m3,
            self.close_to_identity,
        )
    }
}
#[tracked]
pub fn face_forward(v: impl AsExpr<Value = Float3>, n: impl AsExpr<Value = Float3>) -> Expr<Float3> {
    let v = v.as_expr();
    let n = n.as_expr();
    select(v.dot(n) < 0.0, -v, v)
}

#[tracked]
pub fn reflect(w: impl AsExpr<Value = Float3>, n: impl AsExpr<Value = Float3>) -> Expr<Float3> {
    let w = w.as_expr();
    let n = n.as_expr();
    -w + 2.0 * w.dot(n) * n
}
#[tracked]
pub fn refract(
    w: Expr<Float3>,
    n: Expr<Float3>,
    eta: Expr<f32>,
) -> (Expr<bool>, Expr<f32>, Expr<Float3>) {
    // cpu_dbg!(eta);
    let cos_theta_i = w.dot(n);
    let eta = select(cos_theta_i >= (0.0), eta, 1.0 / eta);
    let n = select(cos_theta_i >= (0.0), n, -n);
    let cos_theta_i = cos_theta_i.abs();

    let sin2_theta_i = (1.0 - cos_theta_i.sqr()).max_(0.0);
    let sin2_theta_t = sin2_theta_i / eta.sqr();
    if sin2_theta_t >= 1.0 {
        (false.expr(), eta, Expr::<Float3>::zeroed())
    } else {
        let cos_theta_t = (1.0 - sin2_theta_t).sqrt();
        let wt = -w / eta + (cos_theta_i / eta - cos_theta_t) * n;
        (true.expr(), eta, wt)
    }
    //   // Compute $\cos\,\theta_\roman{t}$ using Snell's law
    //   Float sin2Theta_i = std::max<Float>(0, 1 - Sqr(cosTheta_i));
    //   Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    //   // Handle total internal reflection case
    //   if (sin2Theta_t >= 1)
    //       return false;

    //   Float cosTheta_t = std::sqrt(1 - sin2Theta_t);

    //   *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * Vector3f(n);
}

#[tracked]
pub fn spherical_to_xyz2(
    cos_theta: Expr<f32>,
    sin_theta: Expr<f32>,
    phi: Expr<f32>,
) -> Expr<Float3> {
    Float3::expr(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin())
}
#[tracked]
pub fn spherical_to_xyz(theta: Expr<f32>, phi: Expr<f32>) -> Expr<Float3> {
    let sin_theta = theta.sin();
    Float3::expr(sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin())
}

/// let (theta, phi) = xyz_to_spherical(v);
#[tracked]
pub fn xyz_to_spherical(v: Expr<Float3>) -> (Expr<f32>, Expr<f32>) {
    let phi = v.z.atan2(v.x);
    let theta = v.y.acos();
    (theta, phi)
}

#[tracked]
pub fn invert_phi(v: Expr<Float3>) -> Expr<f32> {
    let phi = v.z.atan2(v.x);
    (phi / (2.0 * PI)).fract()
}
