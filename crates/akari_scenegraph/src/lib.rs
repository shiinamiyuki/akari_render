use std::collections::HashMap;

use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct SceneGraph {
    pub nodes: HashMap<NodeId, Node>,
    pub buffers: HashMap<BufferId, Buffer>,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            buffers: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum Buffer {
    #[serde(rename = "embedded_base64")]
    EmbeddedBase64(String),
    #[serde(rename = "file_path")]
    Path(String),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct BufferView {
    pub buffer: BufferId,
    pub offset: usize,
    pub size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Hash, Eq, PartialEq)]
#[serde(crate = "serde")]
pub struct BufferId(pub u64);

#[derive(Clone, Debug, Serialize, Deserialize, Hash, Eq, PartialEq)]
#[serde(crate = "serde")]
pub struct NodeId(pub u64);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct NodeInput {
    pub node: NodeId,
    pub socket: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct Node {
    pub name: String,
    pub kind: String,
    pub inputs: HashMap<String, NodeInput>,
}
