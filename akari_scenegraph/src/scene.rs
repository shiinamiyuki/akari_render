use base64::Engine;
use rayon::prelude::*;

use crate::{shader::ShaderGraph, *};
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
#[serde(tag = "type")]
pub enum Transform {
    #[serde(rename = "trs")]
    TRS(TRS),
    #[serde(rename = "matrix")]
    Matrix([[f32; 4]; 4]),
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
#[serde(tag = "type")]
pub enum Camera {
    #[serde(rename = "perspective")]
    Perspective(PerspectiveCamera),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointLight {}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Light {
    #[serde(rename = "point")]
    Point(PointLight),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub geometry: NodeRef<Geometry>,
    pub transform: Transform,
    pub material: NodeRef<Material>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Material {
    pub shader: ShaderGraph,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub camera: Option<Camera>,
    pub instances: Collection<Instance>,
    pub geometries: Collection<Geometry>,
    pub materials: Collection<Material>,
    pub lights: Collection<Light>,
    pub buffers: Collection<Buffer>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Buffer {
    #[serde(rename = "binary")]
    EmbeddedBinary { data: Vec<u8> },
    #[serde(rename = "base64")]
    EmbeddedBase64 { data: String },
    #[serde(rename = "path")]
    Path { path: String },
}
impl Buffer {
    pub fn as_binary_data(&self) -> &[u8] {
        match self {
            Buffer::EmbeddedBinary { data } => data,
            _ => {
                panic!("Please embed the buffer before converting to binary")
            }
        }
    }
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        match self {
            Buffer::EmbeddedBinary { data } => {
                assert_eq!(data.len() % std::mem::size_of::<T>(), 0);
                let len = data.len();
                let ptr = data.as_ptr();
                unsafe {
                    std::slice::from_raw_parts(ptr as *const T, len / std::mem::size_of::<T>())
                }
            }
            _ => {
                panic!("Please embed the buffer before converting to binary")
            }
        }
    }
    pub fn load(&mut self, base_path: impl AsRef<Path>) -> std::io::Result<()> {
        let resolve_path = |p: &Path| {
            if p.is_absolute() {
                p.to_owned()
            } else {
                base_path.as_ref().join(p)
            }
        };
        match self {
            Buffer::EmbeddedBinary { .. } | Buffer::EmbeddedBase64 { .. } => {}
            Buffer::Path { path } => {
                let path = resolve_path(Path::new(path));
                let data = std::fs::read(path).map_err(|e| {
                    std::io::Error::new(e.kind(), format!("failed to read buffer: {}", e))
                })?;
                *self = Buffer::EmbeddedBinary { data };
            }
        }
        Ok(())
    }

    /// write contents to file
    /// update self to Buffer::Path
    pub fn write_to_file(&mut self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let mut file = std::fs::File::create(path)?;
        match self {
            Buffer::EmbeddedBinary { data } => {
                file.write_all(data)?;
            }
            Buffer::EmbeddedBase64 { data } => {
                let mut buffer = Vec::new();
                base64::engine::general_purpose::STANDARD_NO_PAD
                    .decode_vec(data, &mut buffer)
                    .unwrap();
                file.write_all(&buffer)?;
            }
            Buffer::Path { path: uri } => {
                let from_path = std::fs::canonicalize(uri)?;
                std::fs::copy(from_path, path)?;
            }
        }
        *self = Buffer::Path {
            path: path.to_str().unwrap().to_string(),
        };
        Ok(())
    }
    pub fn into_binary_inplace(&mut self) {
        match self {
            Buffer::EmbeddedBinary { .. } => {}
            Buffer::EmbeddedBase64 { data } => {
                *self = Buffer::EmbeddedBinary {
                    data: {
                        let mut buffer = Vec::new();
                        base64::engine::general_purpose::STANDARD_NO_PAD
                            .decode_vec(data, &mut buffer)
                            .unwrap();
                        buffer
                    },
                };
            }
            Buffer::Path { .. } => {
                panic!("Please embed the buffer before converting to binary")
            }
        }
    }
    pub fn into_base64_inplace(&mut self) {
        match self {
            Buffer::EmbeddedBinary { data } => {
                *self = Buffer::EmbeddedBase64 {
                    data: {
                        let mut s = String::new();
                        base64::engine::general_purpose::STANDARD_NO_PAD
                            .encode_string(data, &mut s);
                        s
                    },
                };
            }
            Buffer::EmbeddedBase64 { .. } => {}
            Buffer::Path { .. } => {
                panic!("Please embed the buffer before converting to base64")
            }
        }
    }
    pub fn from_vec<T: Copy>(data: Vec<T>) -> Self {
        // todo: optimize!
        Buffer::EmbeddedBinary {
            data: unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<T>(),
                )
            }
            .to_vec(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    #[serde(rename = "srgb")]
    SRgb,
    #[serde(rename = "aces")]
    ACEScg,
    #[serde(rename = "spectral")]
    Spectral,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImageExtenisionMode {
    #[serde(rename = "repeat")]
    Repeat,
    #[serde(rename = "clip")]
    Clip,
    #[serde(rename = "mirror")]
    Mirror,
    #[serde(rename = "extend")]
    Extend,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImageInterpolationMode {
    #[serde(rename = "nearest")]
    Nearest,
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "cubic")]
    Cubic,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ImageFormat {
    #[serde(rename = "png")]
    Png,
    #[serde(rename = "jpeg")]
    Jpeg,
    #[serde(rename = "tiff")]
    Tiff,
    #[serde(rename = "exr")]
    OpenExr,
    #[serde(rename = "float")]
    Float,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Image {
    pub data: NodeRef<Buffer>,
    pub format: ImageFormat,
    pub colorspace: ColorSpace,
    pub extension: ImageExtenisionMode,
    pub interpolation: ImageInterpolationMode,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mesh {
    pub vertices: NodeRef<Buffer>,
    pub normals: Option<NodeRef<Buffer>>,
    pub indices: NodeRef<Buffer>,
    pub uvs: Option<NodeRef<Buffer>>,
    pub tangents: Option<NodeRef<Buffer>>,
    // pub bitangent_signs: Option<Buffer>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Geometry {
    #[serde(rename = "mesh")]
    Mesh(Mesh),
}

impl Scene {
    pub fn new() -> Self {
        Self {
            camera: None,
            instances: Collection::new(),
            geometries: Collection::new(),
            materials: Collection::new(),
            lights: Collection::new(),
            buffers: Collection::new(),
        }
    }
    fn check_valid_buffer(&self, buffer: &NodeRef<Buffer>) {
        assert!(self.buffers.contains_key(buffer));
    }
    /// Embeds/loads all buffers into the scene.
    pub fn load(&mut self, base_path: impl AsRef<Path>) -> std::io::Result<()> {
        let base_path = base_path.as_ref();
        self.buffers
            .par_iter_mut()
            .try_for_each(|(_, buffer)| buffer.load(&base_path))?;
        Ok(())
    }
    pub fn add_buffer(&mut self, name: Option<String>, buffer: Buffer) -> NodeRef<Buffer> {
        let name = name.unwrap_or_else(|| format!("buffer_{}", self.buffers.len()));
        let node_ref = self.buffers.new_ref(Some(name));
        self.buffers.insert(node_ref.clone(), buffer);
        node_ref
    }
    pub fn add_mesh(
        &mut self,
        name: Option<String>,
        vertices: NodeRef<Buffer>,
        indices: NodeRef<Buffer>,
        normals: Option<NodeRef<Buffer>>,
        uvs: Option<NodeRef<Buffer>>,
        tangents: Option<NodeRef<Buffer>>,
    ) -> NodeRef<Geometry> {
        self.check_valid_buffer(&vertices);
        self.check_valid_buffer(&indices);
        if let Some(normals) = normals.as_ref() {
            self.check_valid_buffer(normals);
        }
        if let Some(uvs) = uvs.as_ref() {
            self.check_valid_buffer(uvs);
        }
        if let Some(tangents) = tangents.as_ref() {
            self.check_valid_buffer(tangents);
        }
        let name = name.unwrap_or_else(|| format!("mesh_{}", self.geometries.len()));
        let node_ref = self.geometries.new_ref(Some(name));
        self.geometries.insert(
            node_ref.clone(),
            Geometry::Mesh(Mesh {
                vertices,
                indices,
                normals,
                uvs,
                tangents,
            }),
        );
        node_ref
    }
    pub fn save_json(
        &mut self,
        path: impl AsRef<Path>,
        buffer_in_separate_dir: bool,
    ) -> std::io::Result<()> {
        let abs_path = std::fs::canonicalize(path)?;
        let parent_dir = abs_path.parent().unwrap();
        let buffer_dir = if buffer_in_separate_dir {
            let buffer_dir = parent_dir.join("buffers");
            std::fs::create_dir_all(&buffer_dir)?;
            buffer_dir
        } else {
            parent_dir.to_owned()
        };
        for (r, b) in self.buffers.inner_mut() {
            let filename = format!("{}.bin", r.id);
            // check if filename is valid

            let path = buffer_dir.join(&filename);
            let path = std::fs::canonicalize(path).map_err(|e| {
                std::io::Error::new(e.kind(), format!("invalid filename: {}", filename))
            })?;
            if !path.starts_with(&buffer_dir) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid filename: {}", filename),
                ));
            }

            b.write_to_file(path)?;
        }
        // save scene to file
        let scene_json = serde_json::to_string_pretty(self)?;
        std::fs::write(abs_path, scene_json)
    }
}
