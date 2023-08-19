use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct NodeId(String);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeLink {
    pub from: NodeId,
    pub from_socket: String,
    pub to: NodeId,
    pub to_socket: String,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocketOut {
    pub name: String,
    pub links: Vec<NodeLink>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NodeKind {
    #[serde(rename = "node")]
    Node { kind: String },
    #[serde(rename = "input")]
    Input { name: String },
    #[serde(rename = "output")]
    Output { name: String },
    #[serde(rename = "group")]
    Group { graph: NodeGraph },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub name: NodeId,
    pub kind: NodeKind,
    pub inputs: Vec<SocketIn>,
    pub outputs: Vec<SocketOut>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeGraph {
    pub nodes: HashMap<NodeId, Node>,
}
impl NodeGraph {
    pub fn inline_groups(&mut self) {}
}
pub trait NodeProxy {
    fn from_node(graph: &NodeGraph, node: &Node) -> Self;
    fn to_node(&self, graph: &NodeGraph) -> Node;
    fn category() -> &'static str;
}
#[derive(Clone, Debug)]
pub struct NodeProxyRef<T:CategoryProxy>{
    _marker: std::marker::PhantomData<T>,
    link: Option<NodeLink>,
}
#[derive(Clone, Debug)]
pub enum NodeProxyInput<T: Debug + Clone> {
    Value(T),
    Node(NodeLink),
}
pub trait CategoryProxy {}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Enum {
    pub name: String,
    pub variants: Vec<String>,
}
pub trait EnumProxy {
    fn from_str(s: &str) -> Self;
    fn to_str(&self) -> &str;
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SocketKind {
    Float,
    Int,
    Bool,
    String,
    Enum(Enum),
    Node(String), // Node category
    List(Box<SocketKind>),
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
