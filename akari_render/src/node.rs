use std::{
    collections::BTreeMap,
    ops::{Deref, Index},
};

use serde::{de::Visitor, ser::SerializeMap, Deserialize, Serialize};

use crate::color::RgbColorSpace;
#[derive(Clone, Debug)]
pub struct Collection<T>(BTreeMap<Ref<T>, T>);
impl<'a, T> IntoIterator for &'a Collection<T> {
    type Item = (&'a Ref<T>, &'a T);
    type IntoIter = std::collections::btree_map::Iter<'a, Ref<T>, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<T> std::ops::Deref for Collection<T> {
    type Target = BTreeMap<Ref<T>, T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> std::ops::DerefMut for Collection<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<T> Serialize for Collection<T>
where
    T: Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (key, value) in self.iter() {
            map.serialize_entry(&key.id, value)?;
        }
        map.end()
    }
}
struct CollectionVisitor<T> {
    phantom: std::marker::PhantomData<T>,
}
impl<'de, T: Deserialize<'de>> Visitor<'de> for CollectionVisitor<T> {
    type Value = Collection<T>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a map with string keys and values of type T")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut collection = Collection(BTreeMap::new());
        while let Some((key, value)) = map.next_entry::<String, T>()? {
            collection.insert(
                Ref {
                    id: key,
                    phantom: std::marker::PhantomData,
                },
                value,
            );
        }
        Ok(collection)
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Collection<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(CollectionVisitor {
            phantom: std::marker::PhantomData,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerspectiveCamera {
    pub transform: Transform,
    pub fov: f32,
    pub focal_distance: f32,
    pub fstop: f32,
    pub sensor_width: u32,
    pub sensor_height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Camera {
    Perspective(PerspectiveCamera),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Light {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub camera: Camera,
    pub instances: Collection<Instance>,
    pub geometries: Collection<Geometry>,
    pub materials: Collection<Material>,
    pub lights: Collection<Light>,
    pub images: Collection<Image>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Buffer {
    Internal(Vec<u8>),
    External(String),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: Buffer,
    pub normals: Buffer,
    pub uvs: Buffer,
    pub indices: Buffer,
    pub tangents: Buffer,
    pub bitangent_signs: Buffer,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Geometry {
    Mesh(Mesh),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ref<T> {
    pub id: String,
    phantom: std::marker::PhantomData<T>,
}
impl<T> PartialEq for Ref<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for Ref<T> {}
impl<T> PartialOrd for Ref<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}
impl<T> Ord for Ref<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
impl<T> std::hash::Hash for Ref<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}
impl<T> Deref for Ref<T> {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.id
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum CoordinateSystem {
    Akari,
    Blender,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TRS {
    pub translation: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    pub coordinate_system: CoordinateSystem,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Transform {
    TRS(TRS),
    Matrix([[f32; 4]; 4]),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub geometry: Ref<Geometry>,
    pub transform: Transform,
    pub material: Ref<Material>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Material {
    pub surface: Option<ShaderGraph>,
    pub volume: Option<ShaderGraph>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColorSpace {
    Rgb(RgbColorSpace),
    Spectral,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImageExtenisionMode {
    Repeat,
    Clip,
    Mirror,
    Extend,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImageInterpolationMode {
    Nearest,
    Linear,
    Cubic,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Image {
    pub data: Buffer,
    pub colorspace: ColorSpace,
    pub extension: ImageExtenisionMode,
    pub interpolation: ImageInterpolationMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShaderGraph {
    pub out: Ref<shader::Node>,
    pub nodes: Collection<shader::Node>,
}
impl<'a> Index<&'a Ref<shader::Node>> for ShaderGraph {
    type Output = shader::Node;
    fn index(&self, index: &'a Ref<shader::Node>) -> &Self::Output {
        &self.nodes[index]
    }
}
pub mod shader {
    use std::collections::HashSet;

    use super::*;
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub enum Node {
        Float(f32),
        Float3([f32; 3]),
        Float4([f32; 4]),
        Rgb {
            value: [f32; 3],
            colorspace: RgbColorSpace,
        },
        TexImage(Image),
        DiffuseBsdf {
            color: Ref<Node>,
        },
        GlassBsdf {
            color: Ref<Node>,
            ior: f32,
            roughness: Ref<Node>,
        },
        SpectralUplift(Ref<Node>),
        PrincipledBsdf {
            color: Ref<Node>,
            metallic: Ref<Node>,
            roughness: Ref<Node>,
            specular: Ref<Node>,
            clearcoat: Ref<Node>,
            clearcoat_roughness: Ref<Node>,
            ior: f32,
            transmission: Ref<Node>,
            emission: Ref<Node>,
            emission_strength: Ref<Node>,
        },
        Emission {
            emission: Ref<Node>,
            strength: Ref<Node>,
        },
        MixBsdf {
            first: Ref<Node>,
            second: Ref<Node>,
            factor: Ref<Node>,
        },
        ExtractElement {
            node: Ref<Node>,
            field: String,
        },
    }

    pub struct NodeSorter<'a> {
        graph: &'a ShaderGraph,
        sorted: Vec<Ref<Node>>,
        visited: HashSet<Ref<Node>>,
    }
    #[deny(unused_variables)]
    #[deny(dead_code)]
    impl<'a> NodeSorter<'a> {
        pub fn sort(graph: &'a ShaderGraph) -> Vec<Ref<Node>> {
            let mut sorter = Self {
                graph,
                sorted: vec![],
                visited: HashSet::new(),
            };
            sorter.visit(&graph.out);
            sorter.sorted.reverse();
            sorter.sorted
        }
        fn visit(&mut self, node: &Ref<Node>) {
            if self.visited.contains(node) {
                return;
            }
            self.visited.insert(node.clone());
            self.sorted.push(node.clone());
            let node = &self.graph.nodes[node];
            match node {
                Node::Float(_) => {}
                Node::Float3(_) => {}
                Node::Float4(_) => {}
                Node::Rgb { .. } => {}
                Node::TexImage(_) => {}
                Node::DiffuseBsdf { color } => self.visit(color),
                Node::SpectralUplift(rgb) => self.visit(rgb),
                Node::PrincipledBsdf {
                    color,
                    metallic,
                    roughness,
                    specular,
                    clearcoat,
                    clearcoat_roughness,
                    ior: _,
                    transmission,
                    emission,
                    emission_strength,
                } => {
                    self.visit(color);
                    self.visit(metallic);
                    self.visit(roughness);
                    self.visit(specular);
                    self.visit(clearcoat);
                    self.visit(clearcoat_roughness);
                    self.visit(transmission);
                    self.visit(emission);
                    self.visit(emission_strength);
                }
                Node::Emission { emission, strength } => {
                    self.visit(emission);
                    self.visit(strength);
                }
                Node::MixBsdf {
                    first,
                    second,
                    factor,
                } => {
                    self.visit(first);
                    self.visit(second);
                    self.visit(factor);
                }
                Node::ExtractElement { node, field: _ } => self.visit(node),
                Node::GlassBsdf {
                    color,
                    ior: _,
                    roughness,
                } => {
                    self.visit(color);
                    self.visit(roughness);
                }
            }
        }
    }
}
