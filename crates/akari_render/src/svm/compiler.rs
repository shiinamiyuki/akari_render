use std::collections::HashMap;

use super::*;
use crate::{
    color::ColorSpaceId,
    load::{sampler_from_rgb_image_tex_node, ImageKey},
    util::ByteVecBuilder,
};
use akari_scenegraph::{shader::ShaderNode, NodeRef, ShaderGraph, ShaderKind};
#[derive(Clone, Copy)]
pub struct SvmCompileContext<'a> {
    pub images: &'a HashMap<(ImageKey, TextureSampler), usize>,
    pub graph: &'a ShaderGraph,
}
pub struct CompilerDriver {
    shaders: HashMap<ShaderBytecode, u32>,
    shader_data: Vec<u8>,
}

impl CompilerDriver {
    pub fn variant_count(&self) -> u32 {
        self.shaders.len() as u32
    }
    fn push_shader(&mut self, shader: CompiledShader) -> ShaderRef {
        let CompiledShader { bytecode, data } = shader;
        let kind = if self.shaders.contains_key(&bytecode) {
            self.shaders[&bytecode]
        } else {
            let kind = self.shaders.len() as u32;
            self.shaders.insert(bytecode, kind);
            kind
        };
        assert!(self.shader_data.len() % 16 == 0);
        let base_offset = self.shader_data.len();
        self.shader_data.extend_from_slice(&data);
        {
            // align to 16 bytes
            let padding = 16 - (self.shader_data.len() % 16);
            self.shader_data.extend_from_slice(&vec![0; padding]);
        }
        ShaderRef {
            shader_kind: kind,
            data_offset: base_offset.try_into().unwrap(),
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
            shader_data: Vec::with_capacity(65536),
        }
    }
    pub fn upload(self, device: &Device) -> ShaderCollection {
        let Self {
            shader_data,
            shaders,
        } = self;

        let data = device.create_byte_buffer(shader_data.len());
        data.copy_from(&shader_data[..]);
        let kind_to_shader = shaders
            .iter()
            .map(|(k, v)| (v.clone(), (*k).clone()))
            .collect::<HashMap<_, _>>();
        ShaderCollection {
            shader_to_kind: shaders,
            kind_to_shader,
            shader_data: data,
        }
    }
}

struct Compiler<'a> {
    ctx: SvmCompileContext<'a>,
    env: HashMap<NodeRef<ShaderNode>, SvmNodeRef>,
    bytecode: Vec<SvmNode>,
    data: ByteVecBuilder,
}

impl<'a> Compiler<'a> {
    fn new(ctx: SvmCompileContext<'a>) -> Self {
        Self {
            ctx,
            env: HashMap::new(),
            bytecode: vec![],
            data: ByteVecBuilder::new(),
        }
    }
    fn push(&mut self, node: SvmNode) -> SvmNodeRef {
        let index = self.bytecode.len();
        self.bytecode.push(node);
        SvmNodeRef {
            index: index as u32,
        }
    }
    fn push_data<T: Value>(&mut self, value: T) -> SvmConst<T> {
        let offset = self.data.push(value);
        SvmConst {
            offset: offset as u32,
            marker: std::marker::PhantomData,
        }
    }
    fn compile_node(&mut self, node_id: &NodeRef<ShaderNode>) -> SvmNodeRef {
        let node = self.env.get(node_id);
        if let Some(node) = node {
            return *node;
        }
        self._compile_node(node_id);
        self.env[node_id]
    }
    fn _compile_node(&mut self, node_id: &NodeRef<ShaderNode>) {
        let graph = self.ctx.graph;
        let node = &graph.nodes[node_id];
        let node = match node {
            ShaderNode::Float { value } => {
                let node = self.push_data(*value);
                SvmNode::Float(node)
            }
            ShaderNode::Float3 { value } => {
                let node = self.push_data(Float3::new(value[0], value[1], value[2]));
                SvmNode::Float3(node)
            }
            ShaderNode::Rgb { value, colorspace } => {
                let data = self.push_data(Float3::new(value[0], value[1], value[2]));
                let data = self.push(SvmNode::Float3(data));
                SvmNode::RgbTex(SvmRgbTex {
                    rgb: data,
                    colorspace: ColorSpaceId::from_colorspace((*colorspace).into()),
                })
            }
            ShaderNode::Float4 { .. } => todo!(),
            ShaderNode::TexImage { image } => {
                let colorspace = &image.colorspace;
                let data = &image.data;
                let sampler = sampler_from_rgb_image_tex_node(image);
                let key = ImageKey {
                    buffer: data.clone(),
                    format: image.format,
                    extension: image.extension,
                    interpolation: image.interpolation,
                    width: image.width,
                    height: image.height,
                    channels: image.channels,
                };
                let tex_idx = self.ctx.images[&(key, sampler)] as u32;
                let tex_idx = self.push_data(tex_idx);
                SvmNode::RgbImageTex(SvmRgbImageTex {
                    tex_idx,
                    colorspace: ColorSpaceId::from_colorspace((*colorspace).into()),
                })
            }
            ShaderNode::PerlinNoise { .. } => {
                todo!()
            }
            ShaderNode::DiffuseBsdf { color } => {
                let color = self.compile_node(&color);
                SvmNode::DiffuseBsdf(SvmDiffuseBsdf { reflectance: color })
            }
            ShaderNode::SpectralUplift { rgb } => {
                let rgb = self.compile_node(&rgb);
                SvmNode::SpectralUplift(SvmSpectralUplift { rgb })
            }
            ShaderNode::PrincipledBsdf {
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
                preference: _,
            } => {
                let color = self.compile_node(color);
                let metallic = self.compile_node(metallic);
                let roughness = self.compile_node(roughness);
                let specular = self.compile_node(specular);
                let specular_tint = self.compile_node(specular_tint);
                let clearcoat = self.compile_node(clearcoat);
                let clearcoat_roughness = self.compile_node(clearcoat_roughness);
                let transmission = self.compile_node(transmission);
                let emission = self.compile_node(emission);
                let emission_strength = self.compile_node(emission_strength);
                let ior = self.compile_node(ior);
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
            ShaderNode::Emission {
                color: emission,
                strength,
            } => {
                let emission = self.compile_node(&emission);
                let strength = self.compile_node(&strength);
                SvmNode::Emission(SvmEmission {
                    color: emission,
                    strength,
                })
            }
            ShaderNode::GlassBsdf {
                color,
                ior,
                roughness,
            } => {
                let color = self.compile_node(color);
                let ior = self.compile_node(ior);
                let roughness = self.compile_node(&roughness);
                SvmNode::GlassBsdf(SvmGlassBsdf {
                    kr: color,
                    kt: color,
                    eta: ior,
                    roughness,
                })
            }
            ShaderNode::MixBsdf {
                first: _,
                second: _,
                factor: _,
            } => todo!(),
            ShaderNode::Extract { node: _, field: _ } => todo!(),
            ShaderNode::Output { node } => match self.ctx.graph.kind {
                ShaderKind::Surface => {
                    let surface = self.compile_node(&node);
                    SvmNode::MaterialOutput(SvmMaterialOutput { surface })
                }
                _ => {
                    todo!()
                }
            },
            ShaderNode::Math { op, first, second } => todo!(),
        };

        let node_ref = self.push(node);
        self.env.insert(node_id.clone(), node_ref);
    }

    fn compile(ctx: SvmCompileContext<'a>) -> CompiledShader {
        let mut compiler = Self::new(ctx);
        compiler.compile_node(&ctx.graph.output);
        dbg!(&compiler.bytecode);
        CompiledShader {
            bytecode: ShaderBytecode::new(compiler.bytecode),
            data: compiler.data.finish(),
        }
    }
}
