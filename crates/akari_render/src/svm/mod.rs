use akari_common::parking_lot::Mutex;
use akari_scenegraph::{NodeRef, ShaderNode};
use scene_graph::{MappingType, NormalMapSpace, SeparateColorMode};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    sync::Arc,
};

use crate::{heap::MegaHeap, *};

use self::surface::{EmissiveSurface, PreComputedTable, PreComputedTables, SurfaceShader};
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
    pub uv: Option<SvmNodeRef>,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmCheckerboardTex {
    pub vector: Option<SvmNodeRef>,
    pub scale: SvmNodeRef,
    pub color1: SvmNodeRef,
    pub color2: SvmNodeRef,
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
pub struct SvmPlasticBsdf {
    pub kd: SvmNodeRef,
    pub roughness: SvmNodeRef,
    pub eta: SvmNodeRef,
    pub sigma_a: SvmNodeRef,
    pub thickness: SvmNodeRef,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmMetalBsdf {
    pub kd: SvmNodeRef,
    pub roughness: SvmNodeRef,
    pub eta: SvmNodeRef,
    pub sigma_a: SvmNodeRef,
    pub thickness: SvmNodeRef,
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
pub struct SvmNormalMap {
    pub normal: SvmNodeRef,
    pub strength: SvmNodeRef,
    pub space: NormalMapSpace,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmMapping {
    pub ty: MappingType,
    pub vector: SvmNodeRef,
    pub location: SvmNodeRef,
    pub scale: SvmNodeRef,
    pub rotation: SvmNodeRef,
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
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmTexCoords {}
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SvmExtractField {
    pub node: SvmNodeRef,
    pub field: String,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct SeparateColor {
    pub color: SvmNodeRef,
    pub mode: SeparateColorMode,
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
    CheckerBoard(SvmCheckerboardTex),
    SpectralUplift(SvmSpectralUplift),
    Emission(SvmEmission),
    DiffuseBsdf(SvmDiffuseBsdf),
    GlassBsdf(SvmGlassBsdf),
    PlasticBsdf(SvmPlasticBsdf),
    MetalBsdf(SvmMetalBsdf),
    PrincipledBsdf(SvmPrincipledBsdf),
    NormalMap(SvmNormalMap),
    Mapping(SvmMapping),
    ExtractField(SvmExtractField),
    TexCoords(SvmTexCoords),
    SeparateColor(SeparateColor),
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
    pub(crate) precompute_tables: Mutex<HashMap<ColorRepr, PreComputedTables>>,
}

impl Svm {
    pub fn init_precompute_tables(&self, color_repr: ColorRepr) {
        let mut tables = self.precompute_tables.lock();
        if !tables.contains_key(&color_repr) {
            tables.insert(
                color_repr,
                PreComputedTables::init(self.device.clone(), self.heap.clone(), color_repr),
            );
        }
    }
    pub fn get_precompute_tables(
        &self,
        color_repr: ColorRepr,
        name: impl AsRef<str>,
    ) -> PreComputedTable {
        let tables = self.precompute_tables.lock();
        let name = format!("{}.{}", name.as_ref(), color_repr.to_string());
        tables.get(&color_repr).unwrap().get(&name).unwrap().clone()
    }
    pub fn surface_shaders(&self) -> &ShaderCollection {
        &self.surface_shaders
    }
}
