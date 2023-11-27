use std::f32::consts::{FRAC_1_PI, PI};

use crate::*;

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
#[luisa(crate = "luisa")]
#[value_new(pub)]
pub struct PointNormal {
    pub p: Float3,
    pub n: Float3,
}

#[derive(Clone, Copy, Debug, Soa, Value)]
#[luisa(crate = "luisa")]
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
    #[tracked(crate = "luisa")]
    pub fn at(&self, t: Expr<f32>) -> Expr<Float3> {
        self.o + self.d * t
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct Sphere {
    pub center: Float3,
    pub radius: f32,
}

impl SphereExpr {
    #[tracked(crate = "luisa")]
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

#[derive(Clone, Copy, Debug, Soa, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
#[value_new(pub)]
pub struct Frame {
    pub n: Float3,
    pub t: Float3,
    pub s: Float3,
}

impl Frame {
    #[tracked(crate = "luisa")]
    pub fn cos_theta(w: Expr<Float3>) -> Expr<f32> {
        w.z
    }

    #[tracked(crate = "luisa")]
    pub fn cos2_theta(w: Expr<Float3>) -> Expr<f32> {
        let c = Self::cos_theta(w);
        c * c
    }

    #[tracked(crate = "luisa")]
    pub fn abs_cos_theta(w: Expr<Float3>) -> Expr<f32> {
        Self::cos_theta(w).abs()
    }

    #[tracked(crate = "luisa")]
    pub fn sin_theta(w: Expr<Float3>) -> Expr<f32> {
        let c2 = Self::cos2_theta(w);
        (1.0 - c2).max_(0.0).sqrt()
    }

    #[tracked(crate = "luisa")]
    pub fn sin2_theta(w: Expr<Float3>) -> Expr<f32> {
        let c2 = Self::cos2_theta(w);
        (1.0 - c2).max_(0.0)
    }

    #[tracked(crate = "luisa")]
    pub fn tan2_theta(w: Expr<Float3>) -> Expr<f32> {
        Self::sin2_theta(w) / Self::cos2_theta(w)
    }

    #[tracked(crate = "luisa")]
    pub fn tan_theta(w: Expr<Float3>) -> Expr<f32> {
        Self::sin_theta(w) / Self::cos_theta(w)
    }

    #[tracked(crate = "luisa")]
    pub fn sin_phi(w: Expr<Float3>) -> Expr<f32> {
        let sin_theta = Self::sin_theta(w);
        select(
            sin_theta.eq(0.0),
            0.0f32.expr(),
            (w.x / sin_theta).clamp(-1.0.expr(), 1.0.expr()),
        )
    }

    #[tracked(crate = "luisa")]
    pub fn sin2_phi(w: Expr<Float3>) -> Expr<f32> {
        Self::sin_phi(w).sqr()
    }

    #[tracked(crate = "luisa")]
    pub fn cos2_phi(w: Expr<Float3>) -> Expr<f32> {
        Self::cos_phi(w).sqr()
    }

    #[tracked(crate = "luisa")]
    pub fn cos_phi(w: Expr<Float3>) -> Expr<f32> {
        let sin_theta = Self::sin_theta(w);
        select(
            sin_theta.eq(0.0),
            1.0f32.expr(),
            (w.y / sin_theta).clamp(-1.0.expr(), 1.0.expr()),
        )
    }

    #[tracked(crate = "luisa")]
    pub fn same_hemisphere(w1: Expr<Float3>, w2: Expr<Float3>) -> Expr<bool> {
        (w1.z * w2.z) >= (0.0)
    }
}

impl FrameExpr {
    #[tracked(crate = "luisa")]
    pub fn from_n(n: Expr<Float3>) -> Expr<Frame> {
        let t = if n.x.abs().gt(n.y.abs()) {
            Float3::expr(-n.z, 0.0, n.x) / (n.x * n.x + n.z * n.z).sqrt()
        } else {
            Float3::expr(0.0, n.z, -n.y) / (n.y * n.y + n.z * n.z).sqrt()
        };
        let s = n.cross(t);
        Frame::new_expr(n, t, s)
    }
    #[tracked(crate = "luisa")]
    pub fn to_world(&self, v: Expr<Float3>) -> Expr<Float3> {
        self.t * v.x + self.s * v.y + self.n * v.z
    }
    #[tracked(crate = "luisa")]
    pub fn to_local(&self, v: Expr<Float3>) -> Expr<Float3> {
        Float3::expr(self.t.dot(v), self.s.dot(v), self.n.dot(v))
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
#[value_new(pub)]
pub struct AffineTransform {
    pub m: Mat4,
    pub close_to_identity: bool,
}

impl AffineTransform {
    pub fn from_matrix(m: &glam::Mat4) -> Self {
        let close_to_identity = m.abs_diff_eq(glam::Mat4::IDENTITY, 1e-4);
        Self {
            m: (*m).into(),
            close_to_identity,
        }
    }
    pub fn inverse(&self) -> Self {
        Self {
            m: glam::Mat4::from(self.m).inverse().into(),
            close_to_identity: self.close_to_identity,
        }
    }
}

impl AffineTransformExpr {
    #[tracked(crate = "luisa")]
    pub fn transform_point(&self, p: impl AsExpr<Value=Float3>) -> Expr<Float3> {
        let p = p.as_expr();
        if self.close_to_identity {
            p
        } else {
            let q = Float4::expr(p.x, p.y, p.z, 1.0);
            let q = self.m * q;
            q.xyz() / q.w
        }
    }
    #[tracked(crate = "luisa")]
    pub fn transform_vector(&self, v: impl AsExpr<Value=Float3>) -> Expr<Float3> {
        let v = v.as_expr();
        if self.close_to_identity {
            v
        } else {
            (self.m * v.extend(0.0)).xyz()
        }
    }
    #[tracked(crate = "luisa")]
    pub fn transform_normal(&self, n: impl AsExpr<Value=Float3>) -> Expr<Float3> {
        let n = n.as_expr();
        if self.close_to_identity {
            n
        } else {
            let m3 = Mat3::expr(self.m[0].xyz(), self.m[1].xyz(), self.m[2].xyz());
            m3.inverse().transpose() * n
        }
    }
    pub fn inverse(&self) -> Expr<AffineTransform> {
        AffineTransform::new_expr(self.m.inverse(), self.close_to_identity)
    }
}

#[tracked(crate = "luisa")]
pub fn face_forward(
    v: impl AsExpr<Value=Float3>,
    n: impl AsExpr<Value=Float3>,
) -> Expr<Float3> {
    let v = v.as_expr();
    let n = n.as_expr();
    select(v.dot(n) < 0.0, -v, v)
}

#[tracked(crate = "luisa")]
pub fn reflect(w: impl AsExpr<Value=Float3>, n: impl AsExpr<Value=Float3>) -> Expr<Float3> {
    let w = w.as_expr();
    let n = n.as_expr();
    -w + 2.0 * w.dot(n) * n
}

#[tracked(crate = "luisa")]
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

#[tracked(crate = "luisa")]
pub fn spherical_to_xyz2(
    cos_theta: Expr<f32>,
    sin_theta: Expr<f32>,
    phi: Expr<f32>,
) -> Expr<Float3> {
    Float3::expr(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

#[tracked(crate = "luisa")]
pub fn spherical_to_xyz(theta: Expr<f32>, phi: Expr<f32>) -> Expr<Float3> {
    let sin_theta = theta.sin();
    Float3::expr(sin_theta * phi.cos(), sin_theta * phi.sin(), theta.cos())
}

/// let (theta, phi) = xyz_to_spherical(v);
#[tracked(crate = "luisa")]
pub fn xyz_to_spherical(v: Expr<Float3>) -> (Expr<f32>, Expr<f32>) {
    let phi = v.y.atan2(v.x);
    let theta = v.z.acos();
    (theta, phi)
}

#[tracked(crate = "luisa")]
pub fn invert_phi(v: Expr<Float3>) -> Expr<f32> {
    let phi = v.y.atan2(v.x);
    (phi / (2.0 * PI)).fract()
}

pub fn map_to_sphere_host(p: glam::Vec3) -> glam::Vec2 {
    let l = p.length_squared();
    let u;
    let v;
    if l > 0.0 {
        if p.x == 0.0 && p.y == 0.0 {
            u = 0.0;
        } else {
            u = (0.5 - p.x.atan2(p.y)) * FRAC_1_2PI;
        }
        v = 1.0 - (p.z / l.sqrt()).clamp(-1.0, 1.0).acos() * FRAC_1_PI;
    } else {
        u = 0.0;
        v = 0.0;
    }
    glam::Vec2::new(u, v)
}

#[tracked(crate = "luisa")]
pub fn map_to_sphere(p: Expr<Float3>) -> Expr<Float2> {
    let l = p.length_squared();
    let u = 0.0f32.var();
    let v = 0.0f32.var();
    if l > 0.0 {
        if p.x == 0.0 && p.y == 0.0 {
            *u = 0.0;
        } else {
            *u = (0.5 - p.x.atan2(p.y)) * FRAC_1_2PI;
        }
        *v = 1.0 - (p.z / l.sqrt()).clamp(-1.0f32.expr(), 1.0f32.expr()).acos() * FRAC_1_PI;
    } else {
        *u = 0.0;
        *v = 0.0;
    }
    Float2::expr(u, v)
}
