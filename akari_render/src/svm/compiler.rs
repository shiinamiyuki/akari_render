use std::collections::HashMap;

use super::*;
use crate::{
    color::ColorSpaceId,
    load::sampler_from_rgb_image_tex_node,
    scenegraph::{
        shader::{Node, NodeSorter},
        Ref, ShaderGraph,
    },
};
#[derive(Clone, Copy)]
pub struct SvmCompileContext<'a> {
    pub images: &'a HashMap<(String, TextureSampler), usize>,
    pub graph: &'a ShaderGraph,
}
pub struct CompilerDriver {
    shader_hash_to_kind: HashMap<ShaderHash, u32>,
    shaders: HashMap<ShaderHash, CompiledShader>,
    shader_data: Vec<u8>,
}

impl CompilerDriver {
    pub fn variant_count(&self) -> u32 {
        self.shaders.len() as u32
    }
    fn push_data<T: Value>(&mut self, data: T) -> usize {
        let layout = Layout::new::<T>();
        let size = layout.size();
        let align = layout.align();
        if self.shader_data.len() % align != 0 {
            self.shader_data.resize(
                self.shader_data.len() + align - self.shader_data.len() % align,
                0,
            );
            assert_eq!(self.shader_data.len() % align, 0);
        }
        let offset = self.shader_data.len();
        self.shader_data.resize(offset + size, 0);
        unsafe {
            let ptr = self.shader_data.as_mut_ptr().add(offset);
            std::ptr::copy_nonoverlapping(&data as *const T as *const u8, ptr, size);
        }
        offset
    }
    fn push_shader(&mut self, shader: CompiledShader) -> ShaderRef {
        let expected_offsets = &shader.node_offset;
        let mut base_offset = 0;
        assert!(shader.nodes.len() > 0);
        // dbg!(&shader.nodes);
        for (i, node) in shader.nodes.iter().enumerate() {
            let offset = match node {
                SvmNode::Float(n) => self.push_data(*n),
                SvmNode::Float3(n) => self.push_data(*n),
                SvmNode::MakeFloat3(n) => self.push_data(*n),
                SvmNode::RgbTex(n) => self.push_data(*n),
                SvmNode::RgbImageTex(n) => self.push_data(*n),
                SvmNode::SpectralUplift(n) => self.push_data(*n),
                SvmNode::Emission(n) => self.push_data(*n),
                SvmNode::DiffuseBsdf(n) => self.push_data(*n),
                SvmNode::GlassBsdf(n) => self.push_data(*n),
                SvmNode::PrincipledBsdf(n) => self.push_data(*n),
                SvmNode::MaterialOutput(n) => self.push_data(*n),
            };
            if i != 0 {
                assert_eq!(offset - base_offset, expected_offsets[i]);
            } else {
                base_offset = offset;
            }
        }
        let size = self.shader_data.len() - base_offset;
        assert_eq!(size, shader.size);

        let kind = if self.shaders.contains_key(&shader.hash) {
            self.shader_hash_to_kind[&shader.hash]
        } else {
            let kind = self.shaders.len() as u32;
            self.shader_hash_to_kind.insert(shader.hash, kind);
            self.shaders.insert(shader.hash, shader);
            kind
        };
        ShaderRef {
            shader_kind: kind,
            offset: base_offset.try_into().unwrap(),
            size: size.try_into().unwrap(),
        }
    }
    pub fn compile(&mut self, ctx: SvmCompileContext<'_>) -> ShaderRef {
        // dbg!(out);
        let shader = Compiler::compile(ctx);
        self.push_shader(shader)
    }
    pub fn new() -> Self {
        Self {
            shaders: HashMap::new(),
            shader_hash_to_kind: HashMap::new(),
            shader_data: Vec::with_capacity(65536),
        }
    }
    pub fn upload(self, device: &Device) -> ShaderCollection {
        let Self {
            shader_data,
            shader_hash_to_kind,
            shaders,
        } = self;
        assert_eq!(shader_hash_to_kind.len(), shaders.len());
        let data = device.create_byte_buffer(shader_data.len());
        data.copy_from(&shader_data[..]);
        let shaders = shaders
            .into_iter()
            .map(|(hash, shader)| (shader_hash_to_kind[&hash], shader))
            .collect::<HashMap<_, _>>();
        ShaderCollection {
            shader_hash_to_kind,
            shaders,
            shader_data: data,
        }
    }
}

pub(crate) fn shader_hash(nodes: &[SvmNode]) -> ShaderHash {
    let mut hasher = Sha256::new();
    for node in nodes {
        node.hash(&mut hasher);
    }
    let result = hasher.finalize();
    result.into()
}

struct Compiler<'a> {
    ctx: SvmCompileContext<'a>,
    env: HashMap<Ref<Node>, SvmNodeRef>,
    program: Vec<SvmNode>,
}

impl<'a> Compiler<'a> {
    fn new(ctx: SvmCompileContext<'a>) -> Self {
        Self {
            ctx,
            env: HashMap::new(),
            program: vec![],
        }
    }
    fn push(&mut self, node: SvmNode) -> SvmNodeRef {
        let index = self.program.len();
        self.program.push(node);
        SvmNodeRef {
            index: index as u32,
        }
    }
    fn push_float(&mut self, value: f32) -> SvmNodeRef {
        self.push(SvmNode::Float(SvmFloat { value }))
    }

    fn compile_node(&mut self, node_id: &Ref<Node>) {
        let graph = self.ctx.graph;
        let node = &graph.nodes[node_id];
        let node = match node {
            Node::Float(v) => SvmNode::Float(SvmFloat { value: *v as f32 }),
            Node::Float3(v) => SvmNode::Float3(SvmFloat3 { value: *v }),
            Node::Rgb { value, colorspace } => {
                let data = SvmNode::MakeFloat3(SvmMakeFloat3 {
                    x: self.push_float(value[0]),
                    y: self.push_float(value[1]),
                    z: self.push_float(value[2]),
                });
                let data = self.push(data);
                SvmNode::RgbTex(SvmRgbTex {
                    rgb: data,
                    colorspace: ColorSpaceId::from_colorspace(*colorspace),
                })
            }
            Node::Float4(_) => todo!(),
            Node::TexImage(img) => {
                let colorspace = &img.colorspace;
                let path = &img.path;
                let sampler = sampler_from_rgb_image_tex_node(img);
                let tex_idx = self.ctx.images[&(path.clone(), sampler)];
                SvmNode::RgbImageTex(SvmRgbImageTex {
                    tex_idx: tex_idx as u32,
                    colorspace: ColorSpaceId::from_colorspace(match colorspace {
                        scenegraph::ColorSpace::Rgb(rgb) => *rgb,
                        _ => panic!("not implemented"),
                    }),
                })
            }
            Node::PerlinNoise { .. } => {
                todo!()
            }
            Node::DiffuseBsdf { color } => {
                let color = self.get(&color);
                SvmNode::DiffuseBsdf(SvmDiffuseBsdf { reflectance: color })
            }
            Node::SpectralUplift(rgb) => {
                let rgb = self.get(&rgb);
                SvmNode::SpectralUplift(SvmSpectralUplift { rgb })
            }
            Node::PrincipledBsdf {
                color,
                metallic,
                roughness,
                specular,
                specular_tint,
                clearcoat,
                clearcoat_roughness,
                ior,
                transmission,
                emission,
                emission_strength,
            } => {
                let color = self.get(color);
                let metallic = self.get(metallic);
                let roughness = self.get(roughness);
                let specular = self.get(specular);
                let specular_tint = self.get(specular_tint);
                let clearcoat = self.get(clearcoat);
                let clearcoat_roughness = self.get(clearcoat_roughness);
                let transmission = self.get(transmission);
                let emission = self.get(emission);
                let emission_strength = self.get(emission_strength);
                let ior = self.get(ior);
                SvmNode::PrincipledBsdf(SvmPrincipledBsdf {
                    color,
                    metallic,
                    roughness,
                    specular,
                    specular_tint,
                    clearcoat,
                    clearcoat_roughness,
                    eta: ior,
                    transmission,
                    emission,
                    emission_strength,
                })
            }
            Node::Emission {
                color: emission,
                strength,
            } => {
                let emission = self.get(&emission);
                let strength = self.get(&strength);
                SvmNode::Emission(SvmEmission {
                    color: emission,
                    strength,
                })
            }
            Node::GlassBsdf {
                color,
                ior,
                roughness,
            } => {
                let color = self.get(color);
                let ior = self.get(ior);
                let roughness = self.get(&roughness);
                SvmNode::GlassBsdf(SvmGlassBsdf {
                    kr: color,
                    kt: color,
                    eta: ior,
                    roughness,
                })
            }
            Node::MixBsdf {
                first: _,
                second: _,
                factor: _,
            } => todo!(),
            Node::ExtractElement { node: _, field: _ } => todo!(),
            Node::OutputSurface { surface } => {
                let surface = self.get(&surface);
                SvmNode::MaterialOutput(SvmMaterialOutput { surface })
            }
        };

        let node_ref = self.push(node);
        self.env.insert(node_id.clone(), node_ref);
    }

    fn get(&self, node: &Ref<Node>) -> SvmNodeRef {
        self.env[node]
    }

    fn compile(ctx: SvmCompileContext<'a>) -> CompiledShader {
        let mut compiler = Self::new(ctx);
        let sorted = NodeSorter::sort(ctx.graph);
        for node in sorted {
            compiler.compile_node(&node);
        }
        // dbg!(&compiler.program);
        CompiledShader::new(compiler.program)
    }
}
