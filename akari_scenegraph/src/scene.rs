use std::{collections::HashMap, sync::atomic::AtomicU64};

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
    pub buffer_views: Collection<BufferView>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Buffer {
    #[serde(rename = "binary")]
    EmbeddedBinary { data: Vec<u8> },
    #[serde(rename = "base64")]
    EmbeddedBase64 { data: String, length: u64 },
    #[serde(rename = "path")]
    Path { path: String, length: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BufferView {
    pub buffer: NodeRef<Buffer>,
    pub offset: usize,
    pub length: usize,
}
impl Buffer {
    pub fn len(&self) -> usize {
        match self {
            Buffer::EmbeddedBinary { data } => data.len(),
            Buffer::EmbeddedBase64 { length, .. } => *length as usize,
            Buffer::Path { length, .. } => *length as usize,
        }
    }
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
    pub fn embed(&mut self) -> std::io::Result<()> {
        match self {
            Buffer::EmbeddedBinary { .. } | Buffer::EmbeddedBase64 { .. } => {}
            Buffer::Path {
                path,
                length: lengh,
            } => {
                let data = std::fs::read(path).map_err(|e| {
                    std::io::Error::new(e.kind(), format!("failed to read buffer: {}", e))
                })?;
                if data.len() != *lengh as usize {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "buffer size mismatch: expected {}, got {}",
                            lengh,
                            data.len()
                        ),
                    ));
                }
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
        let length = match self {
            Buffer::EmbeddedBinary { data } => {
                file.write_all(data)?;
                data.len() as u64
            }
            Buffer::EmbeddedBase64 { data, .. } => {
                let mut buffer = Vec::new();
                base64::engine::general_purpose::STANDARD_NO_PAD
                    .decode_vec(data, &mut buffer)
                    .unwrap();
                file.write_all(&buffer)?;
                buffer.len() as u64
            }
            Buffer::Path { path: uri, length } => {
                let from_path = std::fs::canonicalize(uri)?;
                std::fs::copy(from_path, path)?;
                *length
            }
        };
        *self = Buffer::Path {
            path: path.to_str().unwrap().to_string(),
            length,
        };
        Ok(())
    }
    pub fn into_binary_inplace(&mut self) {
        match self {
            Buffer::EmbeddedBinary { .. } => {}
            Buffer::EmbeddedBase64 { data, length } => {
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
                let len = data.len();
                *self = Buffer::EmbeddedBase64 {
                    data: {
                        let mut s = String::new();
                        base64::engine::general_purpose::STANDARD_NO_PAD
                            .encode_string(data, &mut s);
                        s
                    },
                    length: len as u64,
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
    pub data: NodeRef<BufferView>,
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
    pub vertices: NodeRef<BufferView>,
    pub indices: NodeRef<BufferView>,
    pub normals: Option<NodeRef<BufferView>>,
    pub uvs: Option<NodeRef<BufferView>>,
    pub tangents: Option<NodeRef<BufferView>>,
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
            buffer_views: Collection::new(),
        }
    }
    fn check_valid_buffer(&self, buffer: &NodeRef<Buffer>) {
        assert!(self.buffers.contains_key(buffer));
    }
    /// Embeds/loads all buffers into the scene.
    pub fn embed(&mut self) -> std::io::Result<()> {
        self.buffers
            .par_iter_mut()
            .try_for_each(|(_, buffer)| buffer.embed())?;
        Ok(())
    }
    pub fn add_buffer(&mut self, name: Option<String>, buffer: Buffer) -> NodeRef<Buffer> {
        let name = name.unwrap_or_else(|| format!("$buf:{}", self.buffers.len()));
        let node_ref = self.buffers.new_ref(Some(name));
        self.buffers.insert(node_ref.clone(), buffer);
        node_ref
    }
    pub fn add_buffer_view_full(
        &mut self,
        name: Option<String>,
        buffer: NodeRef<Buffer>,
    ) -> NodeRef<BufferView> {
        let offset = 0;
        let length = self.buffers[&buffer].len();
        self.add_buffer_view(name, buffer, offset, length)
    }
    pub fn add_buffer_view(
        &mut self,
        name: Option<String>,
        buffer: NodeRef<Buffer>,
        offset: usize,
        length: usize,
    ) -> NodeRef<BufferView> {
        self.check_valid_buffer(&buffer);
        let name = name.unwrap_or_else(|| format!("$buf_view:{}", self.buffer_views.len()));
        let node_ref = self.buffer_views.new_ref(Some(name));
        self.buffer_views.insert(
            node_ref.clone(),
            BufferView {
                buffer,
                offset,
                length,
            },
        );
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
        let name = name.unwrap_or_else(|| format!("$mesh:{}", self.geometries.len()));
        let node_ref = self.geometries.new_ref(Some(name));
        let vertices = self.add_buffer_view_full(None, vertices);
        let indices = self.add_buffer_view_full(None, indices);
        let normals = normals.map(|n| self.add_buffer_view_full(None, n));
        let uvs = uvs.map(|n| self.add_buffer_view_full(None, n));
        let tangents = tangents.map(|n| self.add_buffer_view_full(None, n));
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
    /// compact buffers into one
    /// requires all buffers to be embedded
    pub fn compact(&mut self, name: Option<String>) -> std::io::Result<()> {
        let total_len = AtomicU64::new(0);
        self.buffers
            .par_iter_mut()
            .try_for_each(|(_, buffer)| -> std::io::Result<()> {
                buffer.embed()?;
                buffer.into_binary_inplace();
                total_len.fetch_add(buffer.len() as u64, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            })?;
        let total_len = total_len.load(std::sync::atomic::Ordering::Relaxed);
        let mut data = Vec::with_capacity(total_len as usize);
        let mut offset = 0;
        let data_ptr = data.as_mut_ptr() as *mut u8;
        let mut buf_ref_to_offset = HashMap::new();
        for (buf_ref, buffer) in self.buffers.iter_mut() {
            let len = buffer.len();
            let buffer_ptr = buffer.as_binary_data().as_ptr() as *const u8;
            unsafe {
                std::ptr::copy_nonoverlapping(buffer_ptr, data_ptr.add(offset), len);
            }
            buf_ref_to_offset.insert(buf_ref.clone(), offset);
            offset += len;
        }
        unsafe {
            data.set_len(total_len as usize);
        }
        // remove all buffers
        self.buffers.clear();
        let buffer = Buffer::from_vec(data);
        let buffer = self.add_buffer(name, buffer);
        // now update all buffer views
        self.buffer_views
            .par_iter_mut()
            .for_each(|(_, buffer_view)| {
                let offset = buf_ref_to_offset[&buffer_view.buffer];
                buffer_view.offset = offset;
                buffer_view.buffer = buffer.clone();
            });
        Ok(())
    }
}
/// A scene that is memory-mapped to disk.
pub struct MmapScene {}
