use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem::transmute;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NodeId(pub String);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeLink {
    pub from: NodeId,
    pub from_socket: String,
    pub to: NodeId,
    pub to_socket: String,
    pub ty: SocketKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SocketValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Enum(String),
    Node(Option<NodeLink>),
    List(Vec<SocketValue>),
}
#[derive(Debug)]
pub struct NodeProxyError {
    pub msg: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocketIn {
    pub name: String,
    pub value: SocketValue,
}
fn cast<From: Any, To: Any + Clone>(v: From) -> Option<To> {
    let any = &v as &dyn Any;
    any.downcast_ref::<To>().map(|v| v.clone())
}
impl SocketIn {
    pub fn as_enum<T: EnumProxy>(&self) -> Result<T, NodeProxyError> {
        match &self.value {
            SocketValue::Enum(v) => Ok(T::from_str(v)),
            _ => Err(NodeProxyError {
                msg: format!(
                    "Invalid value for socket {}: {:?}; not enum!",
                    self.name, self.value
                ),
            }),
        }
    }
    pub fn as_proxy_input<T: SocketType + 'static>(
        &self,
    ) -> Result<NodeProxyInput<T>, NodeProxyError> {
        let name = &self.name;
        let value = &self.value;
        macro_rules! cast {
            ($v:expr) => {
                cast($v).map_or_else(
                    || {
                        Err(NodeProxyError {
                            msg: format!(
                                "Invalid value for socket {}: {:?}; unable to cast!",
                                name, value
                            ),
                        })
                    },
                    Ok,
                )?
            };
        }
        match &self.value {
            SocketValue::Float(v) => Ok(NodeProxyInput::Value(cast!(*v))),
            SocketValue::Int(v) => Ok(NodeProxyInput::Value(cast!(*v))),
            SocketValue::Bool(v) => Ok(NodeProxyInput::Value(cast!(*v))),
            SocketValue::String(v) => Ok(NodeProxyInput::Value(cast!(v.clone()))),
            SocketValue::Enum(_) => Err(NodeProxyError {
                msg: format!("Enum socket {} should be handled by as_enum", name),
            }),
            SocketValue::Node(link) => {
                if link.is_none() {
                    return Ok(NodeProxyInput::Node(None));
                } else {
                    let link = link.as_ref().unwrap();
                    let ty = T::ty();
                    if link.ty != SocketKind::Node(ty.to_string()) {
                        return Err(NodeProxyError {
                            msg: format!("Invalid link type for socket `{}`: {:?}", name, link.ty),
                        });
                    }
                    Ok(NodeProxyInput::Node(Some(link.clone())))
                }
            }
            SocketValue::List(_) => todo!(),
        }
    }
    pub fn as_proxy_input_list<T: SocketType + 'static>(
        &self,
    ) -> Result<Vec<NodeProxyInput<T>>, NodeProxyError> {
        match &self.value {
            SocketValue::List(list) => list
                .iter()
                .map(|v| {
                    let v = SocketIn {
                        name: self.name.clone(),
                        value: v.clone(),
                    };
                    v.as_proxy_input()
                })
                .collect(),
            _ => Err(NodeProxyError {
                msg: format!("Invalid value for socket {}: {:?}", self.name, self.value),
            }),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocketOut {
    pub name: String,
    pub links: Vec<NodeLink>,
}
impl SocketOut {
    pub fn as_proxy_output<T: SocketType + 'static>(
        &self,
    ) -> Result<NodeProxyOutput<T>, NodeProxyError> {
        let ty = T::ty();
        let links = self
            .links
            .iter()
            .filter(|l| l.ty == SocketKind::Node(ty.to_string()))
            .cloned()
            .collect::<Vec<_>>();
        Ok(NodeProxyOutput {
            _marker: std::marker::PhantomData,
            link: links,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeKind {
    #[serde(rename = "node")]
    Node { ty: String },
    #[serde(rename = "input")]
    Input { name: String },
    #[serde(rename = "output")]
    Output { name: String },
    #[serde(rename = "group")]
    Group { graph: NodeGraph },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub kind: NodeKind,
    pub inputs: HashMap<String, SocketIn>,
    pub outputs: HashMap<String, SocketOut>,
}
impl Node {
    pub fn ty(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Node { ty } => Some(ty),
            _ => None,
        }
    }
    pub fn isa(&self, ty: &str) -> bool {
        match &self.kind {
            NodeKind::Node { ty: c } => c == ty,
            _ => false,
        }
    }
    pub fn input(&self, key: &str) -> Result<&SocketIn, NodeProxyError> {
        self.inputs.get(key).map_or_else(
            || {
                Err(NodeProxyError {
                    msg: format!("Input {} not found", key),
                })
            },
            Ok,
        )
    }
    pub fn output(&self, key: &str) -> Result<&SocketOut, NodeProxyError> {
        self.outputs.get(key).map_or_else(
            || {
                Err(NodeProxyError {
                    msg: format!("Output {} not found", key),
                })
            },
            Ok,
        )
    }
    pub fn proxy<T: NodeProxy>(&self, graph: &NodeGraph) -> Result<T, NodeProxyError> {
        T::from_node(graph, self)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGraph {
    pub nodes: HashMap<NodeId, Node>,
}
impl NodeGraph {
    pub fn inline_groups(&mut self) {}
    pub fn get(&self, id: &String) -> Option<&Node> {
        self.nodes.get(&NodeId(id.clone()))
    }
}
pub trait NodeProxy: Sized {
    fn from_node(graph: &NodeGraph, node: &Node) -> Result<Self, NodeProxyError>;
    // fn to_node(&self, graph: &NodeGraph) -> Node;
    fn ty() -> &'static str;
    fn category() -> &'static str;
}

pub trait SocketType: Clone {
    fn is_primitive() -> bool;
    fn ty() -> &'static str;
}
macro_rules! impl_socket_type {
    ($t:ty) => {
        impl SocketType for $t {
            fn is_primitive() -> bool {
                true
            }
            fn ty() -> &'static str {
                stringify!($t)
            }
        }
    };
}
impl_socket_type!(f64);
impl_socket_type!(i64);
impl_socket_type!(bool);
impl_socket_type!(String);
#[derive(Clone, Debug)]
pub struct NodeProxyOutput<T: SocketType> {
    _marker: std::marker::PhantomData<T>,
    pub(crate) link: Vec<NodeLink>,
}
#[derive(Clone, Debug)]
pub enum NodeProxyInput<T: SocketType> {
    Value(T),
    Node(Option<NodeLink>),
}
impl<T: SocketType> NodeProxyInput<T> {
    pub fn as_value(&self) -> Option<&T> {
        match self {
            NodeProxyInput::Value(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_node(&self) -> Option<&NodeLink> {
        match self {
            NodeProxyInput::Node(v) => v.as_ref(),
            _ => None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Enum {
    pub name: String,
    pub variants: Vec<String>,
}
pub trait EnumProxy {
    fn from_str(s: &str) -> Self;
    fn to_str(&self) -> &str;
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SocketKind {
    Float,
    Int,
    Bool,
    String,
    Enum(String),
    Node(String),                         // SocketType
    List(Box<SocketKind>, Option<usize>), // (inner, Optional<length>)
}
impl SocketKind {
    fn ty(&self) -> &str {
        match self {
            SocketKind::Float => "f64",
            SocketKind::Int => "i64",
            SocketKind::Bool => "bool",
            SocketKind::String => "String",
            SocketKind::Enum(e) => &e,
            SocketKind::Node(t) => t,
            SocketKind::List(t, _) => t.ty(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputSocketDesc {
    pub name: String,
    pub kind: SocketKind,
    pub default: SocketValue,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputSocketDesc {
    pub name: String,
    pub kind: SocketKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeDesc {
    pub name: String,
    pub category: String,
    pub inputs: Vec<InputSocketDesc>,
    pub outputs: Vec<OutputSocketDesc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGraphDesc {
    pub nodes: Vec<NodeDesc>,
    pub enums: Vec<Enum>,
}

pub mod gen;
pub mod parse;
