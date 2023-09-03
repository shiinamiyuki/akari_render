use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
};

use crate::*;
use sha2::{Digest, Sha256};
pub mod compiler;
pub mod eval;
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
pub struct SvmMakeFloat3 {
    pub x: SvmNodeRef,
    pub y: SvmNodeRef,
    pub z: SvmNodeRef,
}
impl SvmMakeFloat3 {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.x.hash(hasher);
        self.y.hash(hasher);
        self.z.hash(hasher);
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmRgbTex {
    pub rgb: SvmNodeRef,
    pub colorspace: u32,
}
impl SvmRgbTex {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.rgb.hash(hasher);
        hasher.update(self.colorspace.to_le_bytes());
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmRgbImageTex {
    pub tex_idx: u32,
    pub colorspace: u32,
}
impl SvmRgbImageTex {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        hasher.update(self.colorspace.to_le_bytes());
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
pub struct SvmSpectralUplift {
    pub rgb: SvmNodeRef,
}
impl SvmSpectralUplift {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.rgb.hash(hasher);
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
pub struct SvmGlassBsdf {
    pub kr: SvmNodeRef,
    pub kt: SvmNodeRef,
    pub roughness: SvmNodeRef,
    pub eta: SvmNodeRef,
}
impl SvmGlassBsdf {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.kr.hash(hasher);
        self.kt.hash(hasher);
        self.roughness.hash(hasher);
        self.eta.hash(hasher);
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmPrincipledBsdf {
    pub color: SvmNodeRef,
    pub metallic: SvmNodeRef,
    pub roughness: SvmNodeRef,
    pub specular: SvmNodeRef,
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
    MakeFloat3(SvmMakeFloat3),
    RgbTex(SvmRgbTex),
    RgbImageTex(SvmRgbImageTex),
    SpectralUplift(SvmSpectralUplift),
    DiffuseBsdf(SvmDiffuseBsdf),
    PrincipledBsdf(SvmPrincipledBsdf),
    MaterialOutput(SvmMaterialOutput),
}
impl SvmNode {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        match self {
            SvmNode::Float(x) => x.hash(hasher),
            SvmNode::Float3(x) => x.hash(hasher),
            SvmNode::MakeFloat3(x) => x.hash(hasher),
            SvmNode::RgbTex(x) => x.hash(hasher),
            SvmNode::RgbImageTex(x) => x.hash(hasher),
            SvmNode::SpectralUplift(x) => x.hash(hasher),
            SvmNode::DiffuseBsdf(x) => x.hash(hasher),
            SvmNode::PrincipledBsdf(x) => x.hash(hasher),
            SvmNode::MaterialOutput(x) => x.hash(hasher),
        }
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct ShaderRef {
    pub shader_kind:u32,
    pub shader_id: u32, // n-th shader of this kind
    pub offset: u32,
    pub size: u32,
}

pub struct CompiledShader {
    pub nodes: Vec<SvmNode>,
}

pub type ShaderHash = [u8; 32];

pub struct Svm {
    device: Device,
    pub(crate) shader_hash_to_kind: HashMap<ShaderHash, u32>,
    pub(crate) shaders: HashMap<u32, CompiledShader>, // kind -> shader
    pub(crate) shader_data: ByteBuffer,
    pub(crate) image_textures: BindlessArray,
}


