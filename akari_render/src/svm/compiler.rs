use std::collections::{HashMap, HashSet};

use akari_nodegraph::{Node, NodeGraph, NodeId, NodeProxyInput, SocketValue};

use super::*;
use crate::{color::ColorSpaceId, *};
#[derive(Clone, Copy)]
pub struct SvmCompileContext<'a> {
    pub images: &'a HashMap<String, usize>,
    pub graph: &'a NodeGraph,
}
pub struct CompilerDriver {
    shaders: HashMap<ShaderHash, CompiledShader>,
}

impl CompilerDriver {
    pub fn compile(&self, out: &NodeId, ctx: SvmCompileContext<'_>) {
        todo!()
    }
}

pub struct NodeSorter<'a> {
    graph: &'a NodeGraph,
    sorted: Vec<NodeId>,
    visited: HashSet<NodeId>,
}

fn shader_hash(nodes: &[NodeId]) -> ShaderHash {
    let mut hasher = Sha256::new();
    for node in nodes {
        hasher.update(node.0.as_bytes());
    }
    let result = hasher.finalize();
    result.into()
}
fn get_colorspace_id(colorspace: nodes::ColorSpace) -> u32 {
    match colorspace {
        nodes::ColorSpace::ACEScg => ColorSpaceId::ACES_CG,
        nodes::ColorSpace::SRGB => ColorSpaceId::SRGB,
    }
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
        for (_, socket) in &node.inputs {
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
                colorspace: get_colorspace_id(rgb.in_colorspace),
            })
        });
        when!(
            nodes::RGBImageTexture,
            |compiler: &mut Compiler, node: &Node| {
                let tex = node
                    .proxy::<nodes::RGBImageTexture>(compiler.graph())
                    .unwrap();
                let tex_idx = compiler.ctx.images[&tex.in_path.as_value().unwrap().clone()];
                SvmNode::RgbImageTex(SvmRgbImageTex {
                    tex_idx: tex_idx as u32,
                    colorspace: get_colorspace_id(tex.in_colorspace),
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
                SvmNode::PrincipledBsdf(SvmPrincipledBsdf {
                    color,
                    metallic,
                    roughness,
                    specular,
                    clearcoat,
                    clearcoat_roughness,
                    transmission,
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
        todo!()
    }
}
