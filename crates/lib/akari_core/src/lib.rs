pub mod film;
pub mod geometry;
pub mod spectrum;
pub use film::*;
pub use geometry::*;
pub mod camera;
pub mod distribution;
pub mod sampler;
pub mod scenegraph;
pub mod texture;
pub use akari_utils as util;
pub mod accel;
pub mod bsdf;
pub mod function;
pub mod interaction;
pub mod light;
pub mod rgb2spec;
pub mod sampling;
pub mod scene;
pub mod shape;
pub mod net;
pub mod spmd;
pub use bson;
pub use sampling::*;
#[macro_use]
pub mod color;
use akari_common::*;
pub use glam::{
    uvec2, uvec3, uvec4, vec2, vec3, vec4, Mat2, Mat3, Mat3A, Mat4, UVec2, UVec3, UVec4, Vec2,
    Vec3, Vec3A, Vec4,
};
use parking_lot::*;
pub use rayon::prelude::*;
pub use spectrum::*;
use std::{
    any::Any,
    ops::{Add, Mul, Sub},
};
pub const PI: f32 = std::f32::consts::PI;
pub const FRAC_1_PI: f32 = std::f32::consts::FRAC_1_PI;
pub const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
pub const FRAC_PI_4: f32 = std::f32::consts::FRAC_PI_4;

#[inline]
pub fn lerp3<T>(v0: T, v1: T, v2: T, uv: Vec2) -> T
where
    T: Mul<f32, Output = T> + Add<Output = T> + Copy,
{
    v0 * (1.0 - uv.x - uv.y) + v1 * uv.x + v2 * uv.y
}

#[inline]
pub fn lerp<T, S>(x: T, y: T, a: S) -> T
where
    T: Sub<Output = T> + Add<Output = T> + Mul<S, Output = T> + Copy,
    S: Copy,
{
    x + (y - x) * a
}
// f:(x,y)->T
#[inline]
pub fn bilinear<F: Fn(usize, usize) -> T, T>(f: F, weights: Vec2) -> T
where
    T: Sub<Output = T> + Mul<f32, Output = T> + Add<Output = T> + Copy,
{
    let x0 = lerp(f(0, 0), f(0, 1), weights.y);
    let x1 = lerp(f(1, 0), f(1, 1), weights.y);
    lerp(x0, x1, weights.x)
}
// f:(x,y,z )->T
#[inline]
pub fn trilinear<F: Fn(usize, usize, usize) -> T, T>(f: F, weights: Vec3) -> T
where
    T: Sub<Output = T> + Mul<f32, Output = T> + Add<Output = T> + Copy,
{
    let f = &f;
    let z0 = bilinear(|x, y| f(x, y, 0), vec2(weights.x, weights.y));
    let z1 = bilinear(|x, y| f(x, y, 1), vec2(weights.x, weights.y));
    lerp(z0, z1, weights.z)
}

pub trait AsAny: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn type_name(&self) -> &'static str;
}
impl<T: Any> AsAny for T {
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

pub fn downcast_ref<U: 'static, T: AsAny + ?Sized>(obj: &T) -> Option<&U> {
    obj.as_any().downcast_ref::<U>()
}

pub fn downcast_mut<U: 'static, T: AsAny + ?Sized>(obj: &mut T) -> Option<&mut U> {
    obj.as_any_mut().downcast_mut::<U>()
}
pub fn find_largest<T, P: FnMut(&T) -> bool>(slice: &[T], pred: P) -> usize {
    let i = slice.partition_point(pred);
    (i - 1).clamp(0, slice.len() - 2)
}

#[macro_export]
macro_rules! cond_dbg {
    ($cond:expr, $t:expr) => {
        if $cond {
            dbg!($t)
        } else {
            $t
        }
    };
}
