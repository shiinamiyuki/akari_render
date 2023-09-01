use crate::{mesh::TriangleMesh, scene::Scene, util::binserde::Decode, *};
use akari_nodegraph::{parse, NodeGraph, NodeGraphDesc, NodeId, NodeProxy};
use lazy_static::lazy_static;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
pub struct SceneLoader {
    device: Device,
    pub images: HashMap<PathBuf, image::DynamicImage>,
    pub meshes: HashMap<NodeId, TriangleMesh>,
    pub graph: NodeGraph,
}
fn load_buffer<T>(path: impl AsRef<Path>) -> Vec<T>
where
    Vec<T>: Decode,
{
    let mut file = File::open(path).unwrap();
    Vec::<T>::decode(&mut file).unwrap()
}
impl SceneLoader {
    pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Scene> {
        let mut loader = Self::new(device, load_scene_graph(path));
        todo!()
    }
    fn get_image<P: AsRef<Path>>(&self, path: P) -> &image::DynamicImage {
        self.images.get(path.as_ref()).unwrap()
    }
    fn new(device: Device, graph: NodeGraph) -> Self {
        let mut meshes = HashMap::new();
        let mut images_to_load: HashSet<PathBuf> = HashSet::new();
        for (id, node) in &graph.nodes {
            let ty = node.ty().unwrap();
            if ty == nodes::RGBImageTexture::ty() {
                let tex = node.proxy::<nodes::RGBImageTexture>(&graph).unwrap();
                let path = tex.in_path.as_value().unwrap();
                let path = PathBuf::from(path);
                let path = path.canonicalize().unwrap();
                images_to_load.insert(path);
            } else if ty == nodes::Mesh::ty() {
                let mesh = node.proxy::<nodes::Mesh>(&graph).unwrap();
                let buffers = mesh
                    .in_buffers
                    .iter()
                    .map(|b| {
                        let from = &graph.nodes[&b.as_node().unwrap().from];
                        let buf = from.proxy::<nodes::Buffer>(&graph).unwrap();
                        let path = buf.in_path.as_value().unwrap().clone();
                        let name = buf.in_name.as_value().unwrap().clone();
                        (name, path)
                    })
                    .collect::<HashMap<_, _>>();
                let vertices = load_buffer::<PackedFloat3>(&buffers["vertices"]);
                let normals = load_buffer::<PackedFloat3>(&buffers["normals"]);
                let indices = load_buffer::<PackedUint3>(&buffers["indices"]);
                let mut uvs = vec![];
                if let Some(uv) = buffers.get("uvs") {
                    uvs = load_buffer::<Float2>(uv);
                }
                let mut tangents = vec![];
                if let Some(tangent) = buffers.get("tangents") {
                    tangents = load_buffer::<PackedFloat3>(tangent);
                }
                let mut bitangent_signs = vec![];
                if let Some(bitangent_sign) = buffers.get("bitangent_signs") {
                    bitangent_signs = load_buffer::<u32>(bitangent_sign);
                }
                let mesh = TriangleMesh {
                    name: mesh.in_name.as_value().unwrap().clone(),
                    vertices,
                    normals,
                    indices,
                    uvs,
                    tangents,
                    bitangent_signs,
                };
                meshes.insert(id.clone(), mesh);
            }
        }
        let images = images_to_load
            .into_par_iter()
            .map(|path| {
                log::info!("Loading image: {}", path.display());
                let img = image::open(&path).unwrap();
                (path, img)
            })
            .collect::<HashMap<_, _>>();
        Self {
            device,
            images,
            meshes,
            graph,
        }
    }
}

lazy_static! {
    static ref SCENE_DESC: NodeGraphDesc =
        serde_json::from_str(include_str!("nodes.json")).unwrap();
}
pub fn load_scene_graph<P: AsRef<Path>>(path: P) -> NodeGraph {
    let path = path.as_ref();
    let mut file = File::open(path).unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();
    parse::parse(&buf, &SCENE_DESC).unwrap()
}
