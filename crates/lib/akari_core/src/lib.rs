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
pub mod interaction;
pub mod light;
pub mod rgb2spec;
pub mod scene;
pub mod shape;
use akari_common::*;
use glam::*;
use parking_lot::*;
use rayon::prelude::*;
pub use spectrum::*;
use std::any::Any;
pub const PI: f32 = std::f32::consts::PI;
pub const FRAC_1_PI: f32 = std::f32::consts::FRAC_1_PI;
pub const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
pub const FRAC_PI_4: f32 = std::f32::consts::FRAC_PI_4;

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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
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
