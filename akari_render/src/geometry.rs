use std::f32::consts::PI;

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
    pub exclude0: Uint2,
    pub exclude1: Uint2,
}

impl RayExpr {
    pub fn at(&self, t: Float) -> Expr<Float3> {
        self.o() + self.d() * t
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct Sphere {
    pub center: Float3,
    pub radius: f32,
}
impl SphereExpr {
    pub fn intersect(&self, ray: Expr<Ray>) -> (Expr<bool>, Expr<f32>) {
        let a = ray.d().length_squared();
        let oc = ray.o() - self.center();
        let b = 2.0 * oc.dot(ray.d());
        let c = oc.length_squared() - self.radius().sqr();
        let delta = b.sqr() - 4.0 * a * c;
        if_!(
            delta.cmplt(0.0),
            { (const_(false), const_(0.0f32)) },
            else,
            {
                let t0 = (-b - delta.sqrt()) / (2.0 * a);
                let t1 = (-b + delta.sqrt()) / (2.0 * a);
                if_!(
                    t0.cmplt(ray.t_max()) & t0.cmpge(ray.t_min()),
                    { (const_(true), t0) },
                    else,
                    {
                        if_!(
                            t1.cmplt(ray.t_max()) & t1.cmpge(ray.t_min()),
                            { (const_(true), t1) },
                            else,
                            { (const_(false), const_(0.0f32)) }
                        )
                    }
                )
            }
        )
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
    pub fn p(&self, bary: Expr<Float2>) -> Expr<Float3> {
        (1.0 - bary.x() - bary.y()) * self.v0 + bary.x() * self.v1 + bary.y() * self.v2
    }
    pub fn n(&self, bary: Expr<Float2>) -> Expr<Float3> {
        ((1.0 - bary.x() - bary.y()) * self.n0 + bary.x() * self.n1 + bary.y() * self.n2)
            .normalize()
    }
    pub fn tangent(&self, bary: Expr<Float2>) -> Expr<Float3> {
        ((1.0 - bary.x() - bary.y()) * self.t0 + bary.x() * self.t1 + bary.y() * self.t2)
            .normalize()
    }
    pub fn bitangent(&self, bary: Expr<Float2>) -> Expr<Float3> {
        ((1.0 - bary.x() - bary.y()) * self.b0 + bary.x() * self.b1 + bary.y() * self.b2)
            .normalize()
    }
    pub fn uv(&self, bary: Expr<Float2>) -> Expr<Float2> {
        (1.0 - bary.x() - bary.y()) * self.uv0 + bary.x() * self.uv1 + bary.y() * self.uv2
    }
    pub fn area(&self) -> Float {
        0.5 * (self.v1 - self.v0).cross(self.v2 - self.v0).length()
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
        let c2 = Self::cos2_theta(w);
        (1.0 - c2).max(0.0).sqrt()
    }
    #[inline]
    pub fn sin2_theta(w: Expr<Float3>) -> Float {
        let c2 = Self::cos2_theta(w);
        (1.0 - c2).max(0.0)
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
            const_(0.0f32),
            (w.x() / sin_theta).clamp(-1.0, 1.0),
        )
    }
    #[inline]
    pub fn sin2_phi(w: Expr<Float3>) -> Float {
        Self::sin_phi(w).sqr()
    }
    #[inline]
    pub fn cos2_phi(w: Expr<Float3>) -> Float {
        Self::cos_phi(w).sqr()
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
    -w + 2.0 * w.dot(n) * n
}
pub fn refract(
    w: Expr<Float3>,
    n: Expr<Float3>,
    eta: Expr<f32>,
) -> (Expr<bool>, Expr<f32>, Expr<Float3>) {
    // cpu_dbg!(eta);
    let cos_theta_i = w.dot(n);
    let eta = select(cos_theta_i.cmpge(0.0), eta, 1.0 / eta);
    let n = select(cos_theta_i.cmpge(0.0), n, -n);
    let cos_theta_i = cos_theta_i.abs();

    let sin2_theta_i = (1.0 - cos_theta_i.sqr()).max(0.0);
    let sin2_theta_t = sin2_theta_i / eta.sqr();
    if_!(sin2_theta_t.cmpge(1.0), {
        (const_(false),eta, Float3Expr::zero())
    }, else {
        let cos_theta_t = (1.0 - sin2_theta_t).sqrt();
        let wt = -w / eta + (cos_theta_i / eta - cos_theta_t) * n;
        (const_(true), eta, wt)
    })
    //   // Compute $\cos\,\theta_\roman{t}$ using Snell's law
    //   Float sin2Theta_i = std::max<Float>(0, 1 - Sqr(cosTheta_i));
    //   Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    //   // Handle total internal reflection case
    //   if (sin2Theta_t >= 1)
    //       return false;

    //   Float cosTheta_t = std::sqrt(1 - sin2Theta_t);

    //   *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * Vector3f(n);
}

pub fn spherical_to_xyz2(
    cos_theta: Expr<f32>,
    sin_theta: Expr<f32>,
    phi: Expr<f32>,
) -> Expr<Float3> {
    make_float3(sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin())
}
pub fn spherical_to_xyz(theta: Expr<f32>, phi: Expr<f32>) -> Expr<Float3> {
    let sin_theta = theta.sin();
    make_float3(sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin())
}

// let (theta, phi) = xyz_to_spherical(v);
pub fn xyz_to_spherical(v: Expr<Float3>) -> (Expr<f32>, Expr<f32>) {
    let phi = v.z().atan2(v.x());
    let theta = v.y().acos();
    (theta, phi)
}
pub fn invert_phi(v: Expr<Float3>) -> Expr<f32> {
    let phi = v.z().atan2(v.x());
    (phi / (2.0 * PI)).fract()
}
