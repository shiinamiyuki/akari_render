use crate::{geometry::Sphere, svm::ShaderRef};

use super::*;
#[derive(Clone, Copy, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct PointLight {
    pub light_id: u32,
    pub sphere: Sphere,
    pub surface: ShaderRef,
}
