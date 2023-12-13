use akari_scenegraph::{NodeRef, ShaderNode};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    sync::Arc,
};

use crate::{heap::MegaHeap, *};

use self::surface::{EmissiveSurface, SurfaceShader};
pub mod compiler;
pub mod eval;
pub mod surface;
pub mod texture;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub struct SvmNodeRef {
    pub index: u32, // relative index
}
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SvmConst<T: Value> {
    pub offset: u32,
    pub marker: std::marker::PhantomData<T>,
}
impl<T: Value> Hash for SvmConst<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.offset.hash(state);
    }
}
impl<T: Value> PartialEq for SvmConst<T> {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset
    }
}
impl<T: Value> Eq for SvmConst<T> {}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmMakeFloat3 {
    pub x: SvmNodeRef,
    pub y: SvmNodeRef,
    pub z: SvmNodeRef,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmRgbTex {
    pub rgb: SvmNodeRef,
    pub colorspace: u32,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmRgbImageTex {
    pub tex_idx: SvmConst<u32>,
    pub colorspace: u32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmSpectralUplift {
    pub rgb: SvmNodeRef,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmDiffuseBsdf {
    pub reflectance: SvmNodeRef,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmGlassBsdf {
    pub kr: SvmNodeRef,
    pub kt: SvmNodeRef,
    pub roughness: SvmNodeRef,
    pub eta: SvmNodeRef,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmEmission {
    pub color: SvmNodeRef,
    pub strength: SvmNodeRef,
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmPrincipledBsdf {
    pub base_color: SvmNodeRef,
    pub metallic: SvmNodeRef,
    pub roughness: SvmNodeRef,
    pub ior: SvmNodeRef,
    pub alpha: SvmNodeRef,
    pub normal: SvmNodeRef,
    pub subsurface_weight: SvmNodeRef,
    pub subsurface_radius: SvmNodeRef,
    pub subsurface_scale: SvmNodeRef,
    // pub subsurface_ior: SvmNodeRef,
    pub subsurface_anisotropy: SvmNodeRef,
    pub specular_ior_level: SvmNodeRef,
    pub specular_tint: SvmNodeRef,
    pub anisotropic: SvmNodeRef,
    pub anisotropic_rotation: SvmNodeRef,
    pub tangent: SvmNodeRef,
    pub transmission_weight: SvmNodeRef,
    pub sheen_weight: SvmNodeRef,
    pub sheen_tint: SvmNodeRef,
    pub coat_weight: SvmNodeRef,
    pub coat_roughness: SvmNodeRef,
    pub coat_ior: SvmNodeRef,
    pub coat_tint: SvmNodeRef,
    pub coat_normal: SvmNodeRef,
    pub emission_color: SvmNodeRef,
    pub emission_strength: SvmNodeRef,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmMaterialOutput {
    pub surface: SvmNodeRef,
}
// Shader Virtual Machine
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SvmNode {
    Float(SvmConst<f32>),
    Float3(SvmConst<Float3>),
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
#[derive(Clone, Copy, Debug, Soa, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct ShaderRef {
    pub shader_kind: u32,
    pub data_offset: u32, // offset in bytes
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ShaderBytecode {
    hash: u64,
    nodes: Vec<SvmNode>,
}

pub(crate) fn shader_hash(nodes: &Vec<SvmNode>) -> u64 {
    let mut hasher = DefaultHasher::new();
    Hash::hash(&nodes, &mut hasher);
    hasher.finish()
}

impl ShaderBytecode {
    pub fn new(nodes: Vec<SvmNode>) -> Self {
        let hash = shader_hash(&nodes);
        Self { nodes, hash }
    }
}

pub struct CompiledShader {
    pub bytecode: ShaderBytecode,
    pub data: Vec<u8>,
}

pub struct ShaderCollection {
    #[allow(dead_code)]
    pub(crate) shader_to_kind: HashMap<ShaderBytecode, u32>,
    pub(crate) kind_to_shader: HashMap<u32, ShaderBytecode>,
    pub(crate) shader_data: ByteBuffer,
}
impl ShaderCollection {
    pub fn variant_count(&self) -> usize {
        self.kind_to_shader.len()
    }
}
pub struct Svm {
    #[allow(dead_code)]
    pub(crate) device: Device,
    pub(crate) surface_shaders: ShaderCollection,
    pub(crate) heap: Arc<MegaHeap>,
}

impl Svm {
    pub fn surface_shaders(&self) -> &ShaderCollection {
        &self.surface_shaders
    }
}
