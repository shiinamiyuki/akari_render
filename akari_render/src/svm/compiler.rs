use std::collections::{HashMap, HashSet};

use akari_nodegraph::{Node, NodeGraph, NodeId, NodeProxyInput, SocketValue};

use super::*;
use crate::{color::ColorSpaceId, *};
#[derive(Clone, Copy)]
pub struct SvmCompileContext<'a> {
    pub images: &'a HashMap<(String, luisa::Sampler), usize>,
    pub graph: &'a NodeGraph,
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
    pub fn compile(&mut self, out: &NodeId, ctx: SvmCompileContext<'_>) -> ShaderRef {
        // dbg!(out);
        let shader = Compiler::compile(out.clone(), ctx);
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

pub struct NodeSorter<'a> {
    graph: &'a NodeGraph,
    sorted: Vec<NodeId>,
    visited: HashSet<NodeId>,
}

pub(crate) fn shader_hash(nodes: &[SvmNode]) -> ShaderHash {
    let mut hasher = Sha256::new();
    for node in nodes {
        node.hash(&mut hasher);
    }
    let result = hasher.finalize();
    result.into()
}

impl<'a> NodeSorter<'a> {
    pub fn sort(graph: &'a NodeGraph, root: NodeId) -> Vec<NodeId> {
        let mut sorter = Self {
            graph,
            sorted: vec![],
            visited: HashSet::new(),
        };
        sorter.visit(&root);
        sorter.sorted.reverse();
        sorter.sorted
    }
    fn visit(&mut self, node: &NodeId) {
        if self.visited.contains(node) {
            return;
        }
        self.visited.insert(node.clone());
        self.sorted.push(node.clone());
        let node = &self.graph.nodes[node];
        let mut keys = node.inputs.keys().cloned().collect::<Vec<_>>();
        keys.sort();
        for key in &keys {
            let socket = &node.inputs[key];
            let v = &socket.value;
            match v {
                SocketValue::Node(Some(link)) => {
                    self.visit(&link.from);
                }
                SocketValue::List(nodes) => {
                    for node in nodes {
                        match node {
                            SocketValue::Node(Some(link)) => {
                                self.visit(&link.from);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

struct Compiler<'a> {
    ctx: SvmCompileContext<'a>,
    env: HashMap<NodeId, SvmNodeRef>,
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
    fn compile_float_socket(&mut self, socket: &NodeProxyInput<f64>) -> SvmNodeRef {
        match socket {
            NodeProxyInput::Value(v) => self.push_float(*v as f32),
            NodeProxyInput::Node(Some(link)) => self.get(&link.from),
            _ => unreachable!("Invalid socket value"),
        }
    }
    fn graph(&self) -> &NodeGraph {
        self.ctx.graph
    }
    fn compile_node(&mut self, node_id: &NodeId) {
        let graph = self.ctx.graph;
        let node = &graph.nodes[node_id];
        let ty = node.ty().unwrap();
        let mut pat_match: HashMap<
            &'static str,
            Option<Box<dyn FnOnce(&mut Self, &Node) -> SvmNode>>,
        > = HashMap::new();
        macro_rules! when {
            ($t:ty, $e:expr) => {
                pat_match.insert(<$t as akari_nodegraph::NodeProxy>::ty(), Some(Box::new($e)));
            };
        }
        when!(nodes::Float, |compiler: &mut Compiler, node: &Node| {
            let node = node.proxy::<nodes::Float>(compiler.graph()).unwrap();
            if let Some(v) = node.in_value.as_value() {
                SvmNode::Float(SvmFloat { value: *v as f32 })
            } else {
                panic!("Float node must have a value");
            }
        });
        when!(nodes::RGB, |compiler: &mut Compiler, node: &Node| {
            let rgb = node.proxy::<nodes::RGB>(compiler.graph()).unwrap();
            let rgb_value = if rgb.in_r.as_value().is_some()
                && rgb.in_g.as_value().is_some()
                && rgb.in_b.as_value().is_some()
            {
                let r = *rgb.in_r.as_value().unwrap() as f32;
                let g = *rgb.in_g.as_value().unwrap() as f32;
                let b = *rgb.in_b.as_value().unwrap() as f32;
                SvmNode::Float3(SvmFloat3 {
                    value: PackedFloat3::new(r, g, b),
                })
            } else {
                let r = compiler.compile_float_socket(&rgb.in_r);
                let g = compiler.compile_float_socket(&rgb.in_g);
                let b = compiler.compile_float_socket(&rgb.in_b);
                SvmNode::MakeFloat3(SvmMakeFloat3 { x: r, y: g, z: b })
            };
            let rgb_value = compiler.push(rgb_value);
            SvmNode::RgbTex(SvmRgbTex {
                rgb: rgb_value,
                colorspace: ColorSpaceId::from_colorspace(rgb.in_colorspace.into()),
            })
        });
        when!(
            nodes::RGBImageTexture,
            |compiler: &mut Compiler, node: &Node| {
                let tex = node
                    .proxy::<nodes::RGBImageTexture>(compiler.graph())
                    .unwrap();
                let path = tex.in_path.as_value().unwrap().clone().clone();
                let sampler = load::sampler_from_rgb_image_tex_node(&tex);
                let tex_idx = compiler.ctx.images[&(path, sampler)];
                SvmNode::RgbImageTex(SvmRgbImageTex {
                    tex_idx: tex_idx as u32,
                    colorspace: ColorSpaceId::from_colorspace(tex.in_colorspace.into()),
                })
            }
        );
        when!(
            nodes::SpectralUplift,
            |compiler: &mut Compiler, node: &Node| {
                let uplift = node
                    .proxy::<nodes::SpectralUplift>(compiler.graph())
                    .unwrap();
                let rgb = compiler.get(&uplift.in_rgb.as_node().unwrap().from);
                SvmNode::SpectralUplift(SvmSpectralUplift { rgb })
            }
        );
        when!(
            nodes::DiffuseBsdf,
            |compiler: &mut Compiler, node: &Node| {
                let diffuse = node.proxy::<nodes::DiffuseBsdf>(compiler.graph()).unwrap();
                let color = &diffuse.in_color.as_node().unwrap().from;
                let color = compiler.get(color);
                SvmNode::DiffuseBsdf(SvmDiffuseBsdf { reflectance: color })
            }
        );
        when!(nodes::Emission, |compiler: &mut Compiler, node: &Node| {
            let emission = node.proxy::<nodes::Emission>(compiler.graph()).unwrap();
            let color = &emission.in_color.as_node().unwrap().from;
            let color = compiler.get(color);
            let strength = compiler.compile_float_socket(&emission.in_strength);
            SvmNode::Emission(SvmEmission { color, strength })
        });
        when!(nodes::GlassBsdf, |compiler: &mut Compiler, node: &Node| {
            let glass = node.proxy::<nodes::GlassBsdf>(compiler.graph()).unwrap();
            let color = &glass.in_color.as_node().unwrap().from;
            let color = compiler.get(color);
            let eta = compiler.compile_float_socket(&glass.in_ior);
            let roughness = compiler.compile_float_socket(&glass.in_roughness);
            SvmNode::GlassBsdf(SvmGlassBsdf {
                kr: color,
                kt: color,
                eta,
                roughness,
            })
        });
        when!(
            nodes::PrincipledBsdf,
            |compiler: &mut Compiler, node: &Node| {
                let principled = node
                    .proxy::<nodes::PrincipledBsdf>(compiler.graph())
                    .unwrap();
                let color = compiler.get(&principled.in_color.as_node().unwrap().from);
                let metallic = compiler.compile_float_socket(&principled.in_metallic);
                let roughness = compiler.compile_float_socket(&principled.in_roughness);
                let clearcoat = compiler.compile_float_socket(&principled.in_clearcoat);
                let clearcoat_roughness =
                    compiler.compile_float_socket(&principled.in_clearcoat_roughness);
                let specular = compiler.compile_float_socket(&principled.in_specular);
                let ior = compiler.compile_float_socket(&principled.in_ior);
                let transmission = compiler.compile_float_socket(&principled.in_transmission);
                let emission = compiler.get(&principled.in_emission.as_node().unwrap().from);
                SvmNode::PrincipledBsdf(SvmPrincipledBsdf {
                    color,
                    metallic,
                    roughness,
                    specular,
                    clearcoat,
                    clearcoat_roughness,
                    transmission,
                    emission,
                    eta: ior,
                })
            }
        );
        let f = pat_match.get_mut(ty).unwrap();
        let f = f.take().unwrap();
        let node = (f)(self, node);
        let node_ref = self.push(node);
        self.env.insert(node_id.clone(), node_ref);
    }

    fn get(&self, node: &NodeId) -> SvmNodeRef {
        self.env[node]
    }

    fn compile(out: NodeId, ctx: SvmCompileContext<'a>) -> CompiledShader {
        let mut compiler = Self::new(ctx);
        let sorted = NodeSorter::sort(ctx.graph, out);
        for node in sorted {
            compiler.compile_node(&node);
        }
        // dbg!(&compiler.program);
        CompiledShader::new(compiler.program)
    }
}
