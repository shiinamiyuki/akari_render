use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use crate::*;
use sha2::{Digest, Sha256};
pub mod compiler;
pub mod exec;
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmNodeRef {
    pub index: u32, // relative index
}
impl SvmNodeRef {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(self.index.to_le_bytes());
    }
}
fn type_id_u64<T: 'static>() -> u64 {
    let mut hasher = DefaultHasher::new();
    std::any::TypeId::of::<T>().hash(&mut hasher);
    hasher.finish()
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmFloat {
    pub value: f32,
}
impl SvmFloat {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmFloat3 {
    pub value: PackedFloat3,
}
impl SvmFloat3 {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmTex2d {
    pub index: u32,
}
impl SvmTex2d {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmDiffuseBsdf {
    pub reflectance: SvmNodeRef,
}
impl SvmDiffuseBsdf {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.reflectance.hash(hasher);
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmPrincipledBsdf {
    pub color: SvmNodeRef,
    pub metallic: SvmNodeRef,
    pub roughness: SvmNodeRef,
    pub clearcoat: SvmNodeRef,
    pub clearcoat_roughness: SvmNodeRef,
    pub eta: SvmNodeRef,
    pub transmission: SvmNodeRef,
}
impl SvmPrincipledBsdf {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.color.hash(hasher);
        self.metallic.hash(hasher);
        self.roughness.hash(hasher);
        self.clearcoat.hash(hasher);
        self.clearcoat_roughness.hash(hasher);
        self.eta.hash(hasher);
        self.transmission.hash(hasher);
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmMaterialOutput {
    pub surface: SvmNodeRef,
}
impl SvmMaterialOutput {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.surface.hash(hasher);
    }
}
// Shader Virtual Machine
#[derive(Clone, Debug)]
pub enum SvmNode {
    Float(SvmFloat),
    Float3(SvmFloat3),
    DiffuseBsdf(SvmDiffuseBsdf),
    PrincipledBsdf(SvmPrincipledBsdf),
    MaterialOutput(SvmMaterialOutput),
}
impl SvmNode {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        match self {
            SvmNode::Float(x) => x.hash(hasher),
            SvmNode::Float3(x) => x.hash(hasher),
            SvmNode::DiffuseBsdf(x) => x.hash(hasher),
            SvmNode::PrincipledBsdf(x) => x.hash(hasher),
            SvmNode::MaterialOutput(x) => x.hash(hasher),
        }
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct ShaderHeader {
    pub shader_id: u32,
    pub size: u32,
}

pub struct CompiledShader {
    pub nodes: Vec<SvmNode>,
}

pub type ShaderHash = [u8; 32];
