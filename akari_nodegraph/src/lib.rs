use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem::transmute;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NodeId(String);

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
    pub fn as_enum<T: EnumProxy>(&self) -> Option<T> {
        match &self.value {
            SocketValue::Enum(v) => Some(T::from_str(v)),
            _ => None,
        }
    }
    pub fn as_proxy_input<T: SocketType + 'static>(&self) -> Option<NodeProxyInput<T>> {
        match &self.value {
            SocketValue::Float(v) => Some(NodeProxyInput::Value(cast(*v)?)),
            SocketValue::Int(v) => Some(NodeProxyInput::Value(cast(*v)?)),
            SocketValue::Bool(v) => Some(NodeProxyInput::Value(cast(*v)?)),
            SocketValue::String(v) => Some(NodeProxyInput::Value(cast(v.clone())?)),
            SocketValue::Enum(_) => None,
            SocketValue::Node(link) => {
                if link.is_none() {
                    return Some(NodeProxyInput::Node(None));
                } else {
                    let link = link.as_ref().unwrap();
                    let ty = T::ty();
                    if link.ty != SocketKind::Node(ty.to_string()) {
                        return None;
                    }
                    Some(NodeProxyInput::Node(Some(link.clone())))
                }
            }
            SocketValue::List(_) => todo!(),
        }
    }
    pub fn as_proxy_input_list<T: SocketType + 'static>(&self) -> Option<Vec<NodeProxyInput<T>>> {
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
                .collect::<Option<Vec<_>>>(),
            _ => None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocketOut {
    pub name: String,
    pub links: Vec<NodeLink>,
}
impl SocketOut {
    pub fn as_proxy_output<T: SocketType + 'static>(&self) -> Option<NodeProxyOutput<T>> {
        let ty = T::ty();
        let links = self
            .links
            .iter()
            .filter(|l| l.ty == SocketKind::Node(ty.to_string()))
            .cloned()
            .collect::<Vec<_>>();
        if links.is_empty() {
            return None;
        }
        Some(NodeProxyOutput {
            _marker: std::marker::PhantomData,
            link: links,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
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
    pub fn input(&self, key: &str) -> Option<&SocketIn> {
        self.inputs.get(key)
    }
    pub fn output(&self, key: &str) -> Option<&SocketOut> {
        self.outputs.get(key)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGraph {
    pub nodes: HashMap<NodeId, Node>,
}
impl NodeGraph {
    pub fn inline_groups(&mut self) {}
}
pub trait NodeProxy: Sized {
    fn from_node(graph: &NodeGraph, node: &Node) -> Option<Self>;
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
    link: Vec<NodeLink>,
}
#[derive(Clone, Debug)]
pub enum NodeProxyInput<T: SocketType> {
    Value(T),
    Node(Option<NodeLink>),
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
    Node(String), // SocketType
    List(Box<SocketKind>),
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
            SocketKind::List(t) => t.ty(),
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
