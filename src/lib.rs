#[cfg(feature = "global_mimalloc")]
use mimalloc::MiMalloc;
#[cfg(feature = "global_mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub use nalgebra as na;
use rayon::prelude::*;
#[macro_use]
pub mod accel;
pub mod api;
pub mod bsdf;
pub mod camera;
pub mod distribution;
pub mod film;
#[cfg(feature = "gpu")]
#[macro_use]
pub mod gpu;
pub mod bidir;
pub mod integrator;
pub mod light;
pub mod ltc;
#[macro_use]
mod nn_v2;
#[cfg(feature = "net")]
pub mod net;
pub mod sampler;
pub mod scene;
pub mod scenegraph;
pub mod shape;
pub mod sobolmat;
pub mod texture;
pub mod util;
pub mod varray;
pub mod wavefront;

#[macro_use]
extern crate bitflags;
pub use glam::{
    uvec2, uvec3, uvec4, vec2, vec3, vec4, Mat2, Mat3, Mat3A, Mat4, UVec2, UVec3, UVec4, Vec2,
    Vec3, Vec3A, Vec4,
};
use parking_lot::RwLock;
use serde::Deserialize;
use serde::Serialize;
use std::any::Any;
use std::fmt;
use std::sync::atomic::AtomicUsize;
use std::{
    collections::HashMap,
    ops::{Index, IndexMut, Mul},
    sync::{
        atomic::{AtomicPtr, AtomicU32, Ordering},
        Arc,
    },
    usize,
};

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
#[derive(Serialize, Deserialize)]
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
type Bounds3f = Aabb;
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

#[derive(Clone, Copy, Debug)]
pub struct RGBSpectrum {
    pub samples: Vec3A,
}

pub fn srgb_to_linear(rgb: Vec3) -> Vec3 {
    let f = |s| -> f32 {
        if s <= 0.04045 {
            s / 12.92
        } else {
            (((s + 0.055) / 1.055) as f32).powf(2.4)
        }
    };
    vec3(f(rgb.x), f(rgb.y), f(rgb.z))
}
pub fn linear_to_srgb(linear: Vec3) -> Vec3 {
    let f = |l: f32| -> f32 {
        if l <= 0.0031308 {
            l * 12.92
        } else {
            l.powf(1.0 / 2.4) * 1.055 - 0.055
        }
    };

    vec3(f(linear.x), f(linear.y), f(linear.z))
}
impl RGBSpectrum {
    pub const N_SAMPLES: usize = 3;
    pub fn from_srgb(rgb: Vec3) -> RGBSpectrum {
        RGBSpectrum {
            samples: srgb_to_linear(rgb).into(),
        }
    }
    pub fn to_srgb(&self) -> Vec3 {
        linear_to_srgb(self.samples.into())
    }
    pub fn to_rgb_linear(&self) -> Vec3 {
        self.samples.into()
    }
    pub fn from_rgb_linear(rgb: Vec3) -> Self {
        Self {
            samples: rgb.into(),
        }
    }
    pub fn zero() -> RGBSpectrum {
        Self {
            samples: Vec3A::ZERO,
        }
    }
    pub fn one() -> Spectrum {
        Self {
            samples: Vec3A::ONE,
        }
    }
    // not necessarily black, but any value that is either black or invalid
    pub fn is_black(&self) -> bool {
        !self.samples.is_finite()
            || self.samples.cmpeq(Vec3A::ZERO).all()
            || self.samples.cmplt(Vec3A::ZERO).any()
        // self.samples.iter().any(|x| !x.is_finite())
        //     || self.samples.iter().all(|x| x == 0.0)
        //     || self.samples.iter().any(|x| x < 0.0)
    }
    pub fn lerp(x: RGBSpectrum, y: RGBSpectrum, a: f32) -> RGBSpectrum {
        x * (1.0 - a) + y * a
    }
}
pub fn lerp3v3(v0: Vec3, v1: Vec3, v2: Vec3, uv: Vec2) -> Vec3 {
    (1.0 - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2
}
pub fn lerp3v2(v0: Vec2, v1: Vec2, v2: Vec2, uv: Vec2) -> Vec2 {
    (1.0 - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2
}
pub fn lerp_scalar(x: f32, y: f32, a: f32) -> f32 {
    x + (y - x) * a
}
impl Index<usize> for RGBSpectrum {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.samples[index]
    }
}
impl IndexMut<usize> for RGBSpectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.samples[index]
    }
}
impl std::ops::Add for RGBSpectrum {
    type Output = RGBSpectrum;
    fn add(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples + rhs.samples,
        }
    }
}
impl std::ops::AddAssign for RGBSpectrum {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl std::ops::MulAssign for RGBSpectrum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl std::ops::MulAssign<f32> for RGBSpectrum {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}
impl std::ops::Mul for Spectrum {
    type Output = Spectrum;
    fn mul(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples * rhs.samples,
        }
    }
}
impl std::ops::Mul<f32> for Spectrum {
    type Output = Spectrum;
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            samples: self.samples * rhs,
        }
    }
}
impl std::ops::Div<f32> for Spectrum {
    type Output = Spectrum;
    fn div(self, rhs: f32) -> Self::Output {
        Self {
            samples: self.samples / rhs,
        }
    }
}
pub type Spectrum = RGBSpectrum;
pub fn hsv_to_rgb(hsv: Vec3) -> Vec3 {
    let h = (hsv[0] / 60.0).floor() as u32;
    let f = hsv[0] / 60.0 - h as f32;
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
pub fn rgb_to_hsl(rgb: Vec3) -> Vec3 {
    let max = rgb.max_element();
    let min = rgb.min_element();
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

pub fn rgb_to_hsv(rgb: Vec3) -> Vec3 {
    let max = rgb.max_element();
    let min = rgb.min_element();
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
pub fn int_bits_to_float(x: i32) -> f32 {
    unsafe { std::mem::transmute(x) }
}
pub fn float_bits_to_int(x: f32) -> i32 {
    unsafe { std::mem::transmute(x) }
}
#[derive(Clone, Copy)]
pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,
    pub tmin: f32,
    pub tmax: f32,
    // pub time: f32,
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
pub const PI: f32 = std::f32::consts::PI;
pub const FRAC_1_PI: f32 = std::f32::consts::FRAC_1_PI;
pub const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
pub const FRAC_PI_4: f32 = std::f32::consts::FRAC_PI_4;
pub fn concentric_sample_disk(u: Vec2) -> Vec2 {
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
pub fn consine_hemisphere_sampling(u: Vec2) -> Vec3 {
    let uv = concentric_sample_disk(u);
    let r = uv.length_squared();
    let h = (1.0 - r).sqrt();
    vec3(uv.x, h, uv.y)
}
pub fn uniform_sphere_sampling(u: Vec2) -> Vec3 {
    let z = 1.0 - 2.0 * u[0];
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u[1];
    vec3(r * phi.cos(), z, r * phi.sin())
}
pub fn uniform_sphere_pdf() -> f32 {
    1.0 / (4.0 * PI)
}
pub fn uniform_sample_triangle(u: Vec2) -> Vec2 {
    let mut uf = (u[0] as f64 * (1u64 << 32) as f64) as u64; // Fixed point
    let mut cx = 0.0 as f32;
    let mut cy = 0.0 as f32;
    let mut w = 0.5 as f32;

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
pub fn parallel_for<F: Fn(usize) -> () + Sync>(count: usize, chunk_size: usize, f: F) {
    let nthreads = rayon::current_num_threads();
    let work_counter = AtomicUsize::new(0);
    rayon::scope(|s| {
        for _ in 0..nthreads {
            s.spawn(|_| loop {
                let work = work_counter.fetch_add(chunk_size, Ordering::Relaxed);
                if work >= count {
                    return;
                }
                for i in work..(work + chunk_size).min(count) {
                    f(i);
                }
            });
        }
    });
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
