
pub use glm::vec2;
pub use glm::vec3;
#[cfg(feature = "global_mimalloc")]
use mimalloc::MiMalloc;
#[cfg(feature = "global_mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub use nalgebra as na;
pub use nalgebra_glm as glm;
use rayon::prelude::*;

pub mod accel;
pub mod api;
pub mod bsdf;
pub mod camera;
pub mod distribution;
pub mod film;
#[cfg(feature = "gpu")]
#[macro_use]
pub mod gpu;
pub mod integrator;
pub mod light;
pub mod ltc;
pub mod sampler;
pub mod scene;
pub mod scenegraph;
pub mod shape;
pub mod sobolmat;
pub mod texture;
pub mod util;

#[macro_use]
extern crate bitflags;
use parking_lot::RwLock;
use std::any::Any;
use std::{
    collections::HashMap,
    ops::{Index, IndexMut, Mul},
    sync::{
        atomic::{AtomicPtr, AtomicU32, Ordering},
        Arc,
    },
    usize,
};
#[cfg(feature = "float_f64")]
pub type Float = f64;

#[cfg(not(feature = "float_f64"))]
pub type Float = f32;

pub type Vec3 = glm::TVec3<Float>;
pub type Vec2 = glm::TVec2<Float>;
pub type Mat4 = glm::TMat4<Float>;
pub type Mat3 = glm::TMat3<Float>;

pub fn uvec2(x: u32, y: u32) -> glm::UVec2 {
    glm::UVec2::new(x, y)
}
pub fn uvec3(x: u32, y: u32, z: u32) -> glm::UVec3 {
    glm::UVec3::new(x, y, z)
}
pub fn profile<F: FnOnce() -> T, T>(f: F) -> (T, f64) {
    let now = std::time::Instant::now();
    let ret = f();
    (ret, now.elapsed().as_secs_f64())
}
pub fn profile_ms<F: FnOnce() -> T, T>(f: F) -> (T, f64) {
    let now = std::time::Instant::now();
    let ret = f();
    (ret, now.elapsed().as_secs_f64() * 1000.0)
}
pub struct AtomicFloat {
    bits: AtomicU32,
}
impl Default for AtomicFloat {
    fn default() -> Self {
        Self::new(0.0)
    }
}
impl AtomicFloat {
    pub fn new(v: f32) -> Self {
        Self {
            bits: AtomicU32::new(bytemuck::cast(v)),
        }
    }
    pub fn load(&self, ordering: Ordering) -> f32 {
        bytemuck::cast(self.bits.load(ordering))
    }
    pub fn store(&self, v: f32, ordering: Ordering) {
        self.bits.store(bytemuck::cast(v), ordering)
    }
    pub fn fetch_add(&self, v: f32, ordering: Ordering) -> f32 {
        let mut oldbits = self.bits.load(ordering);
        loop {
            let newbits: u32 = bytemuck::cast(bytemuck::cast::<u32, f32>(oldbits) + v);
            match self.bits.compare_exchange_weak(
                oldbits,
                newbits,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(x) => oldbits = x,
            }
        }
        bytemuck::cast(oldbits)
    }
}
impl Clone for AtomicFloat {
    fn clone(&self) -> Self {
        Self {
            bits: AtomicU32::new(self.bits.load(Ordering::Relaxed)),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Bound2<T: na::Scalar> {
    pub min: glm::TVec2<T>,
    pub max: glm::TVec2<T>,
}

#[derive(Debug, Clone, Copy)]
pub struct Bound3<T: na::Scalar> {
    pub min: glm::TVec3<T>,
    pub max: glm::TVec3<T>,
}

impl<T> Bound3<T>
where
    T: glm::Number,
{
    pub fn size(&self) -> glm::TVec3<T> {
        self.max - self.min
    }

    pub fn insert_point(&mut self, p: &glm::TVec3<T>) -> Self {
        self.min = glm::min2(&self.min, p);
        self.max = glm::max2(&self.max, p);
        *self
    }
    pub fn insert_box(&mut self, aabb: &Self) -> Self {
        self.min = glm::min2(&self.min, &aabb.min);
        self.max = glm::max2(&self.max, &aabb.max);
        *self
    }
}
type Bounds3f = Bound3<Float>;
impl Bound3<Float> {
    pub fn surface_area(&self) -> Float {
        let s = self.size();
        (s[0] * s[1] + s[1] * s[2] + s[0] * s[2]) * 2.0
    }
    pub fn centroid(&self) -> Vec3 {
        self.min + 0.5 * self.size()
    }
    pub fn diagonal(&self) -> Vec3 {
        self.max - self.min
    }
    pub fn contains(&self, p: &Vec3) -> bool {
        glm::all(&glm::less_than_equal(&p, &self.max))
            && glm::all(&glm::greater_than_equal(&p, &self.min))
    }
}
impl Default for Bound3<Float> {
    fn default() -> Self {
        let inf = Float::INFINITY;
        Self {
            min: vec3(inf, inf, inf),
            max: vec3(-inf, -inf, -inf),
        }
    }
}
impl<T> Bound3<T>
where
    T: glm::Number + na::ClosedDiv,
{
    pub fn offset(&self, p: &glm::TVec3<T>) -> glm::TVec3<T> {
        (p - self.min).component_div(&self.size())
    }
}

#[derive(Clone, Copy)]
pub struct Spectrum {
    pub samples: Vec3,
}
impl From<na::SVector<f32, { Spectrum::N_SAMPLES }>> for Spectrum {
    fn from(v: na::SVector<f32, { Spectrum::N_SAMPLES }>) -> Self {
        Spectrum { samples: v.cast() }
    }
}
pub fn srgb_to_linear(rgb: &Vec3) -> Vec3 {
    let f = |s| -> Float {
        if s <= 0.04045 {
            s / 12.92
        } else {
            (((s + 0.055) / 1.055) as Float).powf(2.4)
        }
    };
    vec3(f(rgb.x), f(rgb.y), f(rgb.z))
}
pub fn linear_to_srgb(linear: &Vec3) -> Vec3 {
    let f = |l: Float| -> Float {
        if l <= 0.0031308 {
            l * 12.92
        } else {
            l.powf(1.0 / 2.4) * 1.055 - 0.055
        }
    };

    vec3(f(linear.x), f(linear.y), f(linear.z))
}
impl Spectrum {
    pub const N_SAMPLES: usize = 3;
    pub fn from_srgb(rgb: &Vec3) -> Spectrum {
        Spectrum {
            samples: srgb_to_linear(rgb),
        }
    }
    pub fn to_srgb(&self) -> Vec3 {
        linear_to_srgb(&self.samples)
    }
    pub fn to_rgb_linear(&self) -> Vec3 {
        self.samples
    }
    pub fn from_rgb_linear(rgb: &Vec3) -> Self {
        Self { samples: *rgb }
    }
    pub fn zero() -> Spectrum {
        Self {
            samples: glm::zero(),
        }
    }
    pub fn one() -> Spectrum {
        Self {
            samples: vec3(1.0, 1.0, 1.0),
        }
    }

    pub fn is_black(&self) -> bool {
        !glm::all(&glm::greater_than(&self.samples, &glm::zero()))
            || glm::any(&glm::less_than(&self.samples, &glm::zero()))
    }
    pub fn lerp(x: &Spectrum, y: &Spectrum, a: Float) -> Spectrum {
        *x * (1.0 - a) + *y * a
    }
}
pub fn lerp3<const N: usize>(
    v0: &glm::TVec<Float, N>,
    v1: &glm::TVec<Float, N>,
    v2: &glm::TVec<Float, N>,
    uv: &Vec2,
) -> glm::TVec<Float, N> {
    (1.0 - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2
}
impl Index<usize> for Spectrum {
    type Output = Float;
    fn index(&self, index: usize) -> &Self::Output {
        &self.samples[index]
    }
}
impl IndexMut<usize> for Spectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.samples[index]
    }
}
impl std::ops::Add for Spectrum {
    type Output = Spectrum;
    fn add(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples + rhs.samples,
        }
    }
}
impl std::ops::AddAssign for Spectrum {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl std::ops::MulAssign for Spectrum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl std::ops::MulAssign<Float> for Spectrum {
    fn mul_assign(&mut self, rhs: Float) {
        *self = *self * rhs;
    }
}
impl std::ops::Mul for Spectrum {
    type Output = Spectrum;
    fn mul(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples.component_mul(&rhs.samples),
        }
    }
}
impl std::ops::Mul<Float> for Spectrum {
    type Output = Spectrum;
    fn mul(self, rhs: Float) -> Self::Output {
        Self {
            samples: self.samples * rhs,
        }
    }
}
impl std::ops::Div<Float> for Spectrum {
    type Output = Spectrum;
    fn div(self, rhs: Float) -> Self::Output {
        Self {
            samples: self.samples / rhs,
        }
    }
}
pub fn hsv_to_rgb(hsv: &Vec3) -> Vec3 {
    let h = (hsv[0] / 60.0).floor() as u32;
    let f = hsv[0] / 60.0 - h as Float;
    let p = hsv[2] * (1.0 - hsv[1]);
    let q = hsv[2] * (1.0 - f * hsv[1]);
    let t = hsv[2] * (1.0 - (1.0 - f) * hsv[1]);
    let v = hsv[2];
    let (r, g, b) = match h {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        5 => (v, p, q),
        _ => unreachable!(),
    };
    vec3(r, g, b)
}
pub fn rgb_to_hsl(rgb: &Vec3) -> Vec3 {
    let max = glm::comp_max(rgb);
    let min = glm::comp_min(rgb);
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let h = {
        if max == min {
            0.0
        } else if max == r && g >= b {
            60.0 * (g - b) / (max - min)
        } else if max == r && g < b {
            60.0 * (g - b) / (max - min) + 360.0
        } else if max == g {
            60.0 * (b - r) / (max - min) + 120.0
        } else if max == b {
            60.0 * (r - g) / (max - min) + 240.0
        } else {
            unreachable!()
        }
    };
    let l = 0.5 * (max + min);
    let s = {
        if l == 0.0 || max == min {
            0.0
        } else if 0.0 < l || l <= 0.5 {
            (max - min) / (2.0 * l)
        } else if l > 0.5 {
            (max - min) / (2.0 - 2.0 * l)
        } else {
            unreachable!()
        }
    };
    vec3(h, s, l)
}

pub fn rgb_to_hsv(rgb: &Vec3) -> Vec3 {
    let max = glm::comp_max(rgb);
    let min = glm::comp_min(rgb);
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let h = {
        if max == min {
            0.0
        } else if max == r && g >= b {
            60.0 * (g - b) / (max - min)
        } else if max == r && g < b {
            60.0 * (g - b) / (max - min) + 360.0
        } else if max == g {
            60.0 * (b - r) / (max - min) + 120.0
        } else if max == b {
            60.0 * (r - g) / (max - min) + 240.0
        } else {
            unreachable!()
        }
    };
    let v = max;
    let s = {
        if max == 0.0 {
            0.0
        } else {
            (max - min) / max
        }
    };
    vec3(h, s, v)
}
#[derive(Clone, Copy)]
pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,
    pub tmin: Float,
    pub tmax: Float,
    // pub time: Float,
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
fn offset_ray(p: &Vec3, n: &Vec3) -> Vec3 {
    let of_i = glm::IVec3::new(
        (int_scale() * n.x) as i32,
        (int_scale() * n.y) as i32,
        (int_scale() * n.z) as i32,
    );
    let p_i = vec3(
        glm::int_bits_to_float(
            glm::float_bits_to_int(p.x) + if p.x < 0.0 { -of_i.x } else { of_i.x },
        ),
        glm::int_bits_to_float(
            glm::float_bits_to_int(p.y) + if p.y < 0.0 { -of_i.y } else { of_i.y },
        ),
        glm::int_bits_to_float(
            glm::float_bits_to_int(p.z) + if p.z < 0.0 { -of_i.z } else { of_i.z },
        ),
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
    pub fn spawn(o: &Vec3, d: &Vec3) -> Self {
        Self {
            o: *o,
            d: *d,
            tmin: 0.0,
            tmax: Float::INFINITY,
        }
    }

    #[cfg(feature = "float_f64")]
    pub fn offset_along_normal(&self, n: &Vec3) -> Self {
        Self {
            tmin: self.tmin,
            ..*self
        }
    }
    #[cfg(not(feature = "float_f64"))]
    pub fn offset_along_normal(&self, n: &Vec3) -> Self {
        let p = offset_ray(&self.o, &if glm::dot(&self.d, n) > 0.0 { *n } else { -n });
        let diff = glm::length(&(p - self.o)) / glm::length(&self.d);
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
    pub fn spawn_to(p1: &Vec3, p2: &Vec3) -> Self {
        let len = glm::length(&(p1 - p2));
        let mut ray = Self::spawn(&p1, &glm::normalize(&(p2 - p1)));
        ray.tmax = len;
        ray
    }
    pub fn spawn_to_offseted(p1: &Vec3, p2: &Vec3, n1: &Vec3, n2: &Vec3) -> Self {
        let d = glm::normalize(&(p2 - p1));
        let p1 = offset_ray(p1, &if glm::dot(&d, n1) > 0.0 { *n1 } else { -n1 });
        let p2 = offset_ray(p2, &if glm::dot(&d, n2) > 0.0 { -n2 } else { *n2 });
        let mut ray = Self::spawn(&p1, &p1);
        let len = glm::length(&(p1 - p2));
        ray.tmax = len;
        ray
    }
    pub fn at(&self, t: Float) -> Vec3 {
        self.o + t * self.d
    }
}
#[allow(non_snake_case)]
#[derive(Clone, Copy)]
pub struct Frame {
    pub N: Vec3,
    pub B: Vec3,
    pub T: Vec3,
}
impl Frame {
    pub fn from_normal(normal: &Vec3) -> Self {
        let tangent = if normal.x.abs() > normal.y.abs() {
            glm::normalize(&vec3(-normal.z, 0.0, normal.x))
        } else {
            glm::normalize(&vec3(0.0, normal.z, -normal.y))
        };
        Self {
            N: *normal,
            T: tangent,
            B: glm::normalize(&glm::cross(normal, &tangent)),
        }
    }
    pub fn to_local(&self, v: &Vec3) -> Vec3 {
        vec3(
            glm::dot(&v, &self.T),
            glm::dot(&v, &self.N),
            glm::dot(&v, &self.B),
        )
    }
    pub fn to_world(&self, v: &Vec3) -> Vec3 {
        self.T * v.x + self.N * v.y + self.B * v.z
    }
}
#[derive(Clone, Copy)]
pub struct Transform {
    m4: Mat4,
    inv_m4: Option<Mat4>,
    m3: Mat3,
    inv_m3: Option<Mat3>,
}
impl Transform {
    pub fn inverse(&self) -> Option<Transform> {
        Some(Self {
            m4: self.inv_m4?,
            inv_m4: Some(self.m4),
            m3: self.inv_m3?,
            inv_m3: Some(self.m3),
        })
    }
    pub fn identity() -> Self {
        Self {
            m4: glm::identity(),
            inv_m4: Some(glm::identity()),
            m3: glm::identity(),
            inv_m3: Some(glm::identity()),
        }
    }
    pub fn from_matrix(m: &Mat4) -> Self {
        let m3 = glm::mat4_to_mat3(&m);
        Self {
            m4: *m,
            inv_m4: m.try_inverse(),
            m3,
            inv_m3: m3.try_inverse(),
        }
    }
    pub fn transform_point(&self, p: &Vec3) -> Vec3 {
        let q = self.m4 * glm::TVec4::<Float>::new(p.x, p.y, p.z, 1.0);
        vec3(q.x, q.y, q.z) / q.w
    }
    pub fn transform_vector(&self, v: &Vec3) -> Vec3 {
        self.m3 * v
    }
    pub fn transform_normal(&self, n: &Vec3) -> Vec3 {
        self.inv_m3.unwrap().transpose() * n
    }
}
impl Mul for Transform {
    type Output = Transform;
    fn mul(self, rhs: Transform) -> Self::Output {
        Self {
            m4: self.m4 * rhs.m4,
            inv_m4: if let (Some(a), Some(b)) = (self.inv_m4, rhs.inv_m4) {
                Some(a * b)
            } else {
                None
            },
            m3: self.m3 * rhs.m3,
            inv_m3: if let (Some(a), Some(b)) = (self.inv_m3, rhs.inv_m3) {
                Some(a * b)
            } else {
                None
            },
        }
    }
}
const PI: Float = std::f64::consts::PI as Float;
const FRAC_1_PI: Float = std::f64::consts::FRAC_1_PI as Float;
const FRAC_PI_2: Float = std::f64::consts::FRAC_PI_2 as Float;
const FRAC_PI_4: Float = std::f64::consts::FRAC_PI_4 as Float;
pub fn concentric_sample_disk(u: &Vec2) -> Vec2 {
    let u_offset: Vec2 = 2.0 * u - vec2(1.0, 1.0);
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return vec2(0.0, 0.0);
    }

    let (theta, r) = {
        if u_offset.x.abs() > u_offset.y.abs() {
            let r = u_offset.x;
            let theta = FRAC_PI_4 * (u_offset.y / u_offset.x);
            (theta, r)
        } else {
            let r = u_offset.y;
            let theta = FRAC_PI_2 - FRAC_PI_4 * (u_offset.x / u_offset.y);
            (theta, r)
        }
    };
    r * vec2(theta.cos(), theta.sin())
}
pub fn consine_hemisphere_sampling(u: &Vec2) -> Vec3 {
    let uv = concentric_sample_disk(&u);
    let r = glm::dot(&uv, &uv);
    let h = (1.0 - r).sqrt();
    vec3(uv.x, h, uv.y)
}
pub fn uniform_sphere_sampling(u: &Vec2) -> Vec3 {
    let z = 1.0 - 2.0 * u[0];
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u[1];
    vec3(r * phi.cos(), z, r * phi.sin())
}
pub fn uniform_sphere_pdf() -> Float {
    1.0 / (4.0 * PI)
}
pub fn uniform_sample_triangle(u: &Vec2) -> Vec2 {
    let mut uf = (u[0] as f64 * (1u64 << 32) as f64) as u64; // Fixed point
    let mut cx = 0.0 as Float;
    let mut cy = 0.0 as Float;
    let mut w = 0.5 as Float;

    for _ in 0..16 {
        let uu = uf >> 30;
        let flip = (uu & 3) == 0;

        cy += if (uu & 1) == 0 { 1.0 } else { 0.0 } * w;
        cx += if (uu & 2) == 0 { 1.0 } else { 0.0 } * w;

        w *= if flip { -0.5 } else { 0.5 };
        uf <<= 2;
    }

    let b0 = cx + w / 3.0;
    let b1 = cy + w / 3.0;
    vec2(b0, b1)
}
pub fn dir_to_spherical(v: &Vec3) -> Vec2 {
    let theta = v.y.acos();
    let phi = Float::atan2(v.z, v.x) + PI;
    vec2(theta, phi)
}
pub fn spherical_to_uv(v: &Vec2) -> Vec2 {
    vec2(v.x / PI, v.y / (2.0 * PI))
}
pub fn dir_to_uv(v: &Vec3) -> Vec2 {
    spherical_to_uv(&dir_to_spherical(v))
}
pub fn parallel_for<F: Fn(usize) -> () + Sync>(count: usize, chunk_size: usize, f: F) {
    let chunks = (count + chunk_size - 1) / chunk_size;
    (0..chunks).into_par_iter().for_each(|chunk_id| {
        (chunk_id * chunk_size..(chunk_id * chunk_size + chunk_size).min(count)).for_each(|id| {
            f(id);
        });
    });
}
impl Frame {
    pub fn same_hemisphere(u: &Vec3, v: &Vec3) -> bool {
        u.y * v.y > 0.0
    }
    pub fn cos_theta(u: &Vec3) -> Float {
        u.y
    }
    pub fn cos2_theta(u: &Vec3) -> Float {
        u.y * u.y
    }
    pub fn sin2_theta(u: &Vec3) -> Float {
        (1.0 - Self::cos2_theta(u)).clamp(0.0, 1.0)
    }
    pub fn sin_theta(u: &Vec3) -> Float {
        Self::sin2_theta(u).sqrt()
    }
    pub fn tan_theta(u: &Vec3) -> Float {
        Self::sin_theta(u) / Self::cos_theta(u)
    }
    pub fn abs_cos_theta(u: &Vec3) -> Float {
        u.y.abs()
    }
    pub fn cos_phi(u: &Vec3) -> Float {
        let sin = Self::sin_theta(u);
        if sin == 0.0 {
            1.0
        } else {
            (u.x / sin).clamp(-1.0, 1.0)
        }
    }
    pub fn sin_phi(u: &Vec3) -> Float {
        let sin = Self::sin_theta(u);
        if sin == 0.0 {
            0.0
        } else {
            (u.z / sin).clamp(-1.0, 1.0)
        }
    }
    pub fn cos2_phi(u: &Vec3) -> Float {
        Self::cos_phi(u).powi(2)
    }
    pub fn sin2_phi(u: &Vec3) -> Float {
        Self::sin_phi(u).powi(2)
    }
}
pub fn reflect(w: &Vec3, n: &Vec3) -> Vec3 {
    -w + 2.0 * glm::dot(w, n) * n
}
pub struct Intersection<'a> {
    pub shape: Option<&'a dyn shape::Shape>,
    pub uv: Vec2,
    pub texcoords: Vec2,
    pub t: Float,
    pub ng: Vec3,
    pub ns: Vec3,
    pub prim_id: u32,
}

#[derive(Clone, Copy)]
pub struct UnsafePointer<T> {
    p: *mut T,
}
unsafe impl<T> Sync for UnsafePointer<T> {}
unsafe impl<T> Send for UnsafePointer<T> {}
impl<T> UnsafePointer<T> {
    pub fn new(p: *mut T) -> Self {
        Self { p }
    }
    pub unsafe fn as_mut<'a>(&self) -> Option<&'a mut T> {
        self.p.as_mut()
    }
    pub unsafe fn as_ref<'a>(&self) -> Option<&'a T> {
        self.p.as_ref()
    }
    pub unsafe fn offset(&self, count: isize) -> Self {
        Self {
            p: self.p.offset(count),
        }
    }
}

#[derive(Clone, Copy)]
pub struct UnsafeConstPointer<T> {
    p: *const T,
}
unsafe impl<T> Sync for UnsafeConstPointer<T> {}
unsafe impl<T> Send for UnsafeConstPointer<T> {}
impl<T> UnsafeConstPointer<T> {
    pub fn new(p: *const T) -> Self {
        Self { p }
    }
    pub unsafe fn as_ref<'a>(&self) -> Option<&'a T> {
        self.p.as_ref()
    }
    pub unsafe fn offset(&self, count: isize) -> Self {
        Self {
            p: self.p.offset(count),
        }
    }
}

#[macro_use]
mod nn_v2;

pub trait Base: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn type_name(&self) -> &'static str;
}

pub fn downcast_ref<U: 'static, T: Base + ?Sized>(obj: &T) -> Option<&U> {
    obj.as_any().downcast_ref::<U>()
}

pub fn downcast_mut<U: 'static, T: Base + ?Sized>(obj: &mut T) -> Option<&mut U> {
    obj.as_any_mut().downcast_mut::<U>()
}
#[macro_export]
macro_rules! impl_base {
    ($t:ty) => {
        impl Base for $t {
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
            fn type_name(&self) -> &'static str {
                std::any::type_name::<Self>()
            }
        }
    };
}
