use std::ops::Mul;

use akari_utils::{float_bits_to_int, int_bits_to_float};

use crate::*;
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,
    pub tmin: f32,
    pub tmax: f32,
    // pub time: f32,
}
#[derive(Clone, Copy)]
pub struct Ray4 {
    pub o: [Vec4; 3],
    pub d: [Vec4; 3],
    pub tmin: Vec4,
    pub tmax: Vec4,
    // pub time: f32,
}
impl From<[Ray; 4]> for Ray4 {
    fn from(ray: [Ray; 4]) -> Self {
        Ray4 {
            o: [
                vec4(ray[0].o.x, ray[1].o.x, ray[2].o.x, ray[3].o.x),
                vec4(ray[0].o.y, ray[1].o.y, ray[2].o.y, ray[3].o.y),
                vec4(ray[0].o.z, ray[1].o.z, ray[2].o.z, ray[3].o.z),
            ],
            d: [
                vec4(ray[0].d.x, ray[1].d.x, ray[2].d.x, ray[3].d.x),
                vec4(ray[0].d.y, ray[1].d.y, ray[2].d.y, ray[3].d.y),
                vec4(ray[0].d.z, ray[1].d.z, ray[2].d.z, ray[3].d.z),
            ],
            tmin: vec4(ray[0].tmin, ray[1].tmin, ray[2].tmin, ray[3].tmin),
            tmax: vec4(ray[0].tmax, ray[1].tmax, ray[2].tmax, ray[3].tmax),
        }
    }
}
impl Default for Ray {
    fn default() -> Self {
        Self {
            o: Vec3::ZERO,
            d: Vec3::ZERO,
            tmin: 0.0,
            tmax: -f32::INFINITY,
        }
    }
}
fn origin() -> f32 {
    1.0 / 32.0
}
fn float_scale() -> f32 {
    1.0 / 65536.0
}
fn int_scale() -> f32 {
    256.0
}
fn offset_ray(p: Vec3, n: Vec3) -> Vec3 {
    let of_i = glam::ivec3(
        (int_scale() * n.x) as i32,
        (int_scale() * n.y) as i32,
        (int_scale() * n.z) as i32,
    );
    let p_i = vec3(
        int_bits_to_float(float_bits_to_int(p.x) + if p.x < 0.0 { -of_i.x } else { of_i.x }),
        int_bits_to_float(float_bits_to_int(p.y) + if p.y < 0.0 { -of_i.y } else { of_i.y }),
        int_bits_to_float(float_bits_to_int(p.z) + if p.z < 0.0 { -of_i.z } else { of_i.z }),
    );
    vec3(
        if p.x.abs() < origin() {
            p.x + float_scale() * n.x
        } else {
            p_i.x
        },
        if p.y.abs() < origin() {
            p.y + float_scale() * n.y
        } else {
            p_i.y
        },
        if p.z.abs() < origin() {
            p.z + float_scale() * n.z
        } else {
            p_i.z
        },
    )
}

impl Ray {
    pub fn is_invalid(&self) -> bool {
        self.tmax < self.tmin
    }
    pub fn spawn(o: Vec3, d: Vec3) -> Self {
        Self {
            o,
            d,
            tmin: 0.0,
            tmax: f32::INFINITY,
        }
    }
    pub fn offset_along_normal(&self, n: Vec3) -> Self {
        let p = offset_ray(self.o, if self.d.dot(n) > 0.0 { n } else { -n });
        let diff = (p - self.o).length() / self.d.length();
        Self {
            o: p,
            tmax: if self.tmax.is_infinite() {
                self.tmax
            } else {
                self.tmax - diff
            },
            ..*self
        }
    }
    pub fn spawn_to(p1: Vec3, p2: Vec3) -> Self {
        let len = (p1 - p2).length();
        let mut ray = Self::spawn(p1, (p2 - p1).normalize());
        ray.tmax = len;
        ray
    }
    pub fn spawn_to_offseted(p1: Vec3, p2: Vec3, n1: Vec3, n2: Vec3) -> Self {
        let d = (p2 - p1).normalize();
        let p1 = offset_ray(p1, if d.dot(n1) > 0.0 { n1 } else { -n1 });
        let p2 = offset_ray(p2, if d.dot(n2) > 0.0 { -n2 } else { n2 });
        let mut ray = Self::spawn(p1, p1);
        let len = (p1 - p2).length();
        ray.tmax = len;
        ray
    }
    pub fn at(&self, t: f32) -> Vec3 {
        self.o + t * self.d
    }
}

#[derive(Clone, Copy)]
pub struct ReferencePoint {
    pub p: Vec3,
    pub n: Vec3,
}

#[allow(non_snake_case)]
#[derive(Clone, Copy)]
pub struct Frame {
    pub N: Vec3,
    pub B: Vec3,
    pub T: Vec3,
}
impl Frame {
    pub fn from_normal(normal: Vec3) -> Self {
        let tangent = if normal.x.abs() > normal.y.abs() {
            vec3(-normal.z, 0.0, normal.x).normalize()
        } else {
            vec3(0.0, normal.z, -normal.y).normalize()
        };
        Self {
            N: normal,
            T: tangent,
            B: normal.cross(tangent).normalize(),
        }
    }
    pub fn to_local(&self, v: Vec3) -> Vec3 {
        vec3(v.dot(self.T), v.dot(self.N), v.dot(self.B))
    }
    pub fn to_world(&self, v: Vec3) -> Vec3 {
        self.T * v.x + self.N * v.y + self.B * v.z
    }
}
#[derive(Clone, Copy)]
pub struct Transform {
    m4: Mat4,
    inv_m4: Mat4,
    m3: Mat3A,
    inv_m3: Mat3A,
}
impl Transform {
    pub fn inverse(&self) -> Transform {
        Self {
            m4: self.inv_m4,
            inv_m4: self.m4,
            m3: self.inv_m3,
            inv_m3: self.m3,
        }
    }
    pub fn identity() -> Self {
        Self {
            m4: Mat4::IDENTITY,
            inv_m4: Mat4::IDENTITY,
            m3: Mat3A::IDENTITY,
            inv_m3: Mat3A::IDENTITY,
        }
    }
    pub fn from_matrix(m: &Mat4) -> Self {
        let m3 = Mat3A::from_mat4(*m);
        Self {
            m4: *m,
            inv_m4: m.inverse(),
            m3,
            inv_m3: m3.inverse(),
        }
    }
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        let q = self.m4 * vec4(p.x, p.y, p.z, 1.0);
        vec3(q.x, q.y, q.z) / q.w
    }
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        self.m3 * v
    }
    pub fn transform_normal(&self, n: Vec3) -> Vec3 {
        self.inv_m3.transpose() * n
    }
}
impl Mul for Transform {
    type Output = Transform;
    fn mul(self, rhs: Transform) -> Self::Output {
        Self::from_matrix(&self.m4.mul_mat4(&rhs.m4))
    }
}

pub fn dir_to_spherical(v: Vec3) -> Vec2 {
    let theta = v.y.acos();
    let phi = f32::atan2(v.z, v.x) + PI;
    vec2(theta, phi)
}
pub fn spherical_to_uv(v: Vec2) -> Vec2 {
    vec2(v.x / PI, v.y / (2.0 * PI))
}
pub fn dir_to_uv(v: Vec3) -> Vec2 {
    spherical_to_uv(dir_to_spherical(v))
}

impl Frame {
    #[inline]
    pub fn same_hemisphere(u: Vec3, v: Vec3) -> bool {
        u.y * v.y > 0.0
    }
    pub fn cos_theta(u: Vec3) -> f32 {
        u.y
    }
    pub fn cos2_theta(u: Vec3) -> f32 {
        u.y * u.y
    }
    pub fn sin2_theta(u: Vec3) -> f32 {
        (1.0 - Self::cos2_theta(u)).clamp(0.0, 1.0)
    }
    pub fn sin_theta(u: Vec3) -> f32 {
        Self::sin2_theta(u).sqrt()
    }
    pub fn tan_theta(u: Vec3) -> f32 {
        Self::sin_theta(u) / Self::cos_theta(u)
    }
    pub fn abs_cos_theta(u: Vec3) -> f32 {
        u.y.abs()
    }
    pub fn cos_phi(u: Vec3) -> f32 {
        let sin = Self::sin_theta(u);
        if sin == 0.0 {
            1.0
        } else {
            (u.x / sin).clamp(-1.0, 1.0)
        }
    }
    pub fn sin_phi(u: Vec3) -> f32 {
        let sin = Self::sin_theta(u);
        if sin == 0.0 {
            0.0
        } else {
            (u.z / sin).clamp(-1.0, 1.0)
        }
    }
    pub fn cos2_phi(u: Vec3) -> f32 {
        Self::cos_phi(u).powi(2)
    }
    pub fn sin2_phi(u: Vec3) -> f32 {
        Self::sin_phi(u).powi(2)
    }
}
pub fn reflect(w: Vec3, n: Vec3) -> Vec3 {
    -w + 2.0 * w.dot(n) * n
}
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub uv: Vec2,
    pub t: f32,
    pub ng: Vec3,
    pub prim_id: u32,
    pub geom_id: u32,
}
impl RayHit {
    pub fn is_invalid(&self) -> bool {
        self.prim_id == u32::MAX || self.geom_id == u32::MAX
    }
}
impl Default for RayHit {
    fn default() -> Self {
        Self {
            uv: Vec2::ZERO,
            t: -f32::INFINITY,
            ng: Vec3::ZERO,
            prim_id: u32::MAX,
            geom_id: u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Aabb {
    pub fn size(&self) -> Vec3 {
        (self.max - self.min).into()
    }

    pub fn insert_point(&mut self, p: Vec3) -> Self {
        self.min = self.min.min(p.into());
        self.max = self.max.max(p.into());
        *self
    }
    pub fn insert_box(&mut self, aabb: Self) -> Self {
        self.min = self.min.min(aabb.min);
        self.max = self.max.max(aabb.max);
        *self
    }
}

pub type Bounds3f = Aabb;
impl Aabb {
    pub fn surface_area(&self) -> f32 {
        let s = self.size();
        (s[0] * s[1] + s[1] * s[2] + s[0] * s[2]) * 2.0
    }
    pub fn centroid(&self) -> Vec3 {
        Vec3::from(self.min) + 0.5 * self.size()
    }
    pub fn diagonal(&self) -> Vec3 {
        (self.max - self.min).into()
    }
    pub fn contains(&self, p: Vec3) -> bool {
        p.cmple(self.max.into()).all() && p.cmpge(self.min.into()).all()
    }
    pub fn offset(&self, p: Vec3) -> Vec3 {
        (p - Vec3::from(self.min)) / self.size()
    }
}
impl Default for Aabb {
    fn default() -> Self {
        let inf = f32::INFINITY;
        Self {
            min: Vec3A::new(inf, inf, inf),
            max: Vec3A::new(-inf, -inf, -inf),
        }
    }
}
