use std::{fs::File, io::Read, path::Path};

use akari_nodegraph::{parse, NodeGraph, NodeGraphDesc};

pub struct SceneLoader {}
static SCENE_DESC: &'static str = include_str!("nodes.json");
pub fn load_scene_graph<P: AsRef<Path>>(path: P) -> NodeGraph {
    let path = path.as_ref();
    let mut file = File::open(path).unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();
    let desc = serde_json::from_str::<NodeGraphDesc>(SCENE_DESC).unwrap();
    parse::parse(&buf, &desc).unwrap()
}
