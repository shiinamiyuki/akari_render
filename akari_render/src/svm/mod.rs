use std::{
    alloc::Layout,
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
};

use crate::{util::round_to, *};
use luisa::runtime::api::Shader;
use sha2::{Digest, Sha256};

use self::surface::{EmissiveSurface, SurfaceShader};
pub mod compiler;
pub mod eval;
pub mod surface;
pub mod texture;
#[derive(Clone, Copy, Debug, Value, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub struct SvmNodeRef {
    pub index: u32, // relative index
}
impl SvmNodeRef {
    pub const INVALID: Self = Self { index: u32::MAX };
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
    pub value: [f32; 3],
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
pub struct SvmEmission {
    pub color: SvmNodeRef,
    pub strength: SvmNodeRef,
}
impl SvmEmission {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.color.hash(hasher);
        self.strength.hash(hasher);
    }
}
impl SurfaceShader for SvmEmission {
    fn closure(&self, svm_eval: &eval::SvmEvaluator<'_>) -> std::rc::Rc<dyn surface::Surface> {
        let color = svm_eval.eval_color(self.color);
        let strength = svm_eval.eval_float(self.strength);
        std::rc::Rc::new(EmissiveSurface {
            inner: None,
            emission: color * strength,
        })
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct SvmPrincipledBsdf {
    pub color: SvmNodeRef,
    pub metallic: SvmNodeRef,
    pub roughness: SvmNodeRef,
    pub specular: SvmNodeRef,
    pub specular_tint: SvmNodeRef,
    pub clearcoat: SvmNodeRef,
    pub clearcoat_roughness: SvmNodeRef,
    pub eta: SvmNodeRef,
    pub transmission: SvmNodeRef,
    pub emission: SvmNodeRef,
    pub emission_strength: SvmNodeRef,
}
impl SvmPrincipledBsdf {
    pub fn hash<H: Digest>(&self, hasher: &mut H) {
        hasher.update(type_id_u64::<Self>().to_le_bytes());
        self.color.hash(hasher);
        self.metallic.hash(hasher);
        self.roughness.hash(hasher);
        self.specular.hash(hasher);
        self.specular_tint.hash(hasher);
        self.clearcoat.hash(hasher);
        self.clearcoat_roughness.hash(hasher);
        self.eta.hash(hasher);
        self.transmission.hash(hasher);
        self.emission.hash(hasher);
        self.emission_strength.hash(hasher);
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
    Emission(SvmEmission),
    DiffuseBsdf(SvmDiffuseBsdf),
    GlassBsdf(SvmGlassBsdf),
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
            SvmNode::Emission(x) => x.hash(hasher),
            SvmNode::DiffuseBsdf(x) => x.hash(hasher),
            SvmNode::GlassBsdf(x) => x.hash(hasher),
            SvmNode::PrincipledBsdf(x) => x.hash(hasher),
            SvmNode::MaterialOutput(x) => x.hash(hasher),
        }
    }
    pub fn layout(&self) -> Layout {
        match self {
            SvmNode::Float(_) => Layout::new::<SvmFloat>(),
            SvmNode::Float3(_) => Layout::new::<SvmFloat3>(),
            SvmNode::MakeFloat3(_) => Layout::new::<SvmMakeFloat3>(),
            SvmNode::RgbTex(_) => Layout::new::<SvmRgbTex>(),
            SvmNode::RgbImageTex(_) => Layout::new::<SvmRgbImageTex>(),
            SvmNode::SpectralUplift(_) => Layout::new::<SvmSpectralUplift>(),
            SvmNode::Emission(_) => Layout::new::<SvmEmission>(),
            SvmNode::DiffuseBsdf(_) => Layout::new::<SvmDiffuseBsdf>(),
            SvmNode::GlassBsdf(_) => Layout::new::<SvmGlassBsdf>(),
            SvmNode::PrincipledBsdf(_) => Layout::new::<SvmPrincipledBsdf>(),
            SvmNode::MaterialOutput(_) => Layout::new::<SvmMaterialOutput>(),
        }
    }
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct ShaderRef {
    pub shader_kind: u32,
    pub offset: u32, // offset in bytes
    pub size: u32,
}

pub struct CompiledShader {
    pub nodes: Vec<SvmNode>,
    pub node_offset: Vec<usize>,
    pub size: usize,
    pub hash: ShaderHash,
}
impl CompiledShader {
    pub fn new(nodes: Vec<SvmNode>) -> Self {
        let mut node_offset = Vec::new();
        let mut offset = 0;
        for node in &nodes {
            offset = round_to(offset, node.layout().align());
            node_offset.push(offset);
            offset += node.layout().size();
        }
        let hash = compiler::shader_hash(&nodes);
        // {
        //     // debug
        //     // convert hash to string
        //     let mut hash_str = String::new();
        //     for byte in &hash {
        //         hash_str.push_str(&format!("{:02x}", byte));
        //     }
        //     println!("shader hash: {}", hash_str);
        // }
        Self {
            nodes,
            node_offset,
            hash,
            size: offset,
        }
    }
}

pub type ShaderHash = [u8; 32];
pub struct ShaderCollection {
    pub(crate) shader_hash_to_kind: HashMap<ShaderHash, u32>,
    pub(crate) shaders: HashMap<u32, CompiledShader>, // kind -> shader
    pub(crate) shader_data: ByteBuffer,
}
impl ShaderCollection {
    pub fn variant_count(&self) -> usize {
        self.shaders.len()
    }
}
pub struct Svm {
    pub(crate) device: Device,
    pub(crate) surface_shaders: ShaderCollection,
    pub(crate) image_textures: BindlessArray,
}
