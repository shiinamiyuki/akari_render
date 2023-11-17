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
    pub camera: Camera,
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
    Path { uri: String },
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
    pub fn embed(&mut self, base_path: impl AsRef<Path>) -> std::io::Result<()> {
        let resolve_path = |p: &Path| {
            if p.is_absolute() {
                p.to_owned()
            } else {
                base_path.as_ref().join(p)
            }
        };
        match self {
            Buffer::EmbeddedBinary { .. } | Buffer::EmbeddedBase64 { .. } => {}
            Buffer::Path { uri: path } => {
                let path = resolve_path(Path::new(path));
                let data = std::fs::read(path).map_err(|e| {
                    std::io::Error::new(e.kind(), format!("failed to read buffer: {}", e))
                })?;
                *self = Buffer::EmbeddedBinary { data };
            }
        }
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
    /// Embeds/loads all buffers into the scene.
    pub fn embed(&mut self, base_path: impl AsRef<Path>) -> std::io::Result<()> {
        let base_path = base_path.as_ref();
        self.buffers
            .par_iter_mut()
            .try_for_each(|(_, buffer)| buffer.embed(&base_path))?;
        Ok(())
    }
}
