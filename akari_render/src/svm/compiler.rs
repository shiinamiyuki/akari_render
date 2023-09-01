use std::collections::{HashMap, HashSet};

use akari_nodegraph::{Node, NodeGraph, NodeId, SocketValue};

use super::*;
use crate::*;

pub struct CompilerDriver {
    shaders: HashMap<ShaderHash, CompiledShader>,
}

impl CompilerDriver {
    pub fn compile(&self, out: &NodeId, graph: &NodeGraph) {}
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
    graph: &'a NodeGraph,
    env: HashMap<NodeId, SvmNodeRef>,
    program: Vec<SvmNode>,
}
impl<'a> Compiler<'a> {
    fn new(graph: &'a NodeGraph) -> Self {
        Self {
            graph,
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
    fn compile_node(&mut self, node_id: &NodeId) {
        let node = &self.graph.nodes[node_id];
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
        when!(nodes::RGB, |compiler, node| { todo!() });
        when!(
            nodes::DiffuseBsdf,
            |compiler: &mut Compiler, node: &Node| {
                let diffuse = node.proxy::<nodes::DiffuseBsdf>(compiler.graph).unwrap();
                let color = diffuse.in_color.as_node().unwrap();
                SvmNode::DiffuseBsdf(SvmDiffuseBsdf {
                    reflectance: todo!(),
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

    fn compile(out: NodeId, graph: &'a NodeGraph) -> CompiledShader {
        let mut compiler = Self::new(graph);
        let sorted = NodeSorter::sort(graph, out);
        for node in sorted {
            compiler.compile_node(&node);
        }
        todo!()
    }
}
