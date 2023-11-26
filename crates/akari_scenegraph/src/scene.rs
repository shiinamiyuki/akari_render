use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{atomic::AtomicU64, Arc},
};

use crate::base64::Engine;
use crate::memmap2::Mmap;
use rayon::prelude::*;

use crate::{shader::ShaderGraph, *};
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(crate = "serde")]
pub enum CoordinateSystem {
    Akari,
    Blender,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct TRS {
    pub translation: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3],
    pub coordinate_system: CoordinateSystem,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum Transform {
    #[serde(rename = "trs")]
    TRS(TRS),
    #[serde(rename = "matrix")]
    Matrix([[f32; 4]; 4]),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct PerspectiveCamera {
    pub transform: Transform,
    pub fov: f32,
    pub focal_distance: f32,
    pub fstop: f32,
    pub sensor_width: u32,
    pub sensor_height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum Camera {
    #[serde(rename = "perspective")]
    Perspective(PerspectiveCamera),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct PointLight {}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum Light {
    #[serde(rename = "point")]
    Point(PointLight),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct Instance {
    pub geometry: NodeRef<Geometry>,
    pub transform: Transform,
    pub materials: Vec<NodeRef<Material>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct Material {
    pub shader: ShaderGraph,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
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
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum Buffer {
    #[serde(rename = "binary")]
    EmbeddedBinary { data: Vec<u8> },
    #[serde(rename = "base64")]
    EmbeddedBase64 { data: String, length: u64 },
    #[serde(rename = "path")]
    Path { path: String, length: u64 },
    // for internal use only, cannot be transformed into other variants directly
    #[serde(rename = "__unused_slice")]
    Slice { slice: ExtSlice },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(crate = "serde")]
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
            Buffer::Slice { slice } => slice.len(),
        }
    }
    pub unsafe fn as_binary_data(&self) -> &[u8] {
        match self {
            Buffer::EmbeddedBinary { data } => data,
            Buffer::Slice { slice } => unsafe { slice.as_slice() },
            _ => {
                panic!("Please embed the buffer before converting to binary")
            }
        }
    }
    pub unsafe fn as_slice<T: Copy>(&self) -> &[T] {
        let data = self.as_binary_data();
        let len = data.len();
        let ptr = data.as_ptr();
        unsafe { std::slice::from_raw_parts(ptr as *const T, len / std::mem::size_of::<T>()) }
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
            Buffer::Slice { .. } => {}
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
            Buffer::Slice { .. } => {
                panic!("Please convert slice to binary before embedding");
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
                let data = {
                    let mut buffer = Vec::new();
                    base64::engine::general_purpose::STANDARD_NO_PAD
                        .decode_vec(data, &mut buffer)
                        .unwrap();
                    buffer
                };
                assert_eq!(data.len() as u64, *length);
                *self = Buffer::EmbeddedBinary { data };
            }
            Buffer::Path { .. } => {
                panic!("Please embed the buffer before converting to binary")
            }
            Buffer::Slice { .. } => {}
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
            Buffer::Slice { .. } => {
                panic!("Please convert slice to binary manually");
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]#[serde(crate = "serde")]
pub enum ColorSpace {
    #[serde(rename = "srgb")]
    SRgb,
    #[serde(rename = "aces")]
    ACEScg,
    #[serde(rename = "spectral")]
    Spectral,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]#[serde(crate = "serde")]
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
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]#[serde(crate = "serde")]
pub enum ImageInterpolationMode {
    #[serde(rename = "nearest")]
    Nearest,
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "cubic")]
    Cubic,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]#[serde(crate = "serde")]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]#[serde(crate = "serde")]
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

#[derive(Clone, Debug, Serialize, Deserialize)]#[serde(crate = "serde")]
pub struct Mesh {
    pub vertices: NodeRef<BufferView>,
    pub indices: NodeRef<BufferView>,
    pub normals: Option<NodeRef<BufferView>>,
    pub uvs: Option<NodeRef<BufferView>>,
    pub tangents: Option<NodeRef<BufferView>>,
    pub materials: NodeRef<BufferView>,
    // pub bitangent_signs: Option<Buffer>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]#[serde(crate = "serde")]
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
        let name = name.unwrap_or_else(|| format!("buf_{}", self.buffers.len()));
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
        let name = name.unwrap_or_else(|| format!("buf_view_{}", self.buffer_views.len()));
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
        materials: NodeRef<Buffer>,
    ) -> NodeRef<Geometry> {
        self.check_valid_buffer(&vertices);
        self.check_valid_buffer(&indices);
        self.check_valid_buffer(&materials);
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
        let vertices = self.add_buffer_view_full(None, vertices);
        let indices = self.add_buffer_view_full(None, indices);
        let normals = normals.map(|n| self.add_buffer_view_full(None, n));
        let uvs = uvs.map(|n| self.add_buffer_view_full(None, n));
        let tangents = tangents.map(|n| self.add_buffer_view_full(None, n));
        let materials = self.add_buffer_view_full(None, materials);
        self.geometries.insert(
            node_ref.clone(),
            Geometry::Mesh(Mesh {
                vertices,
                indices,
                normals,
                uvs,
                tangents,
                materials,
            }),
        );
        node_ref
    }
    pub fn add_material(&mut self, name: Option<String>, material: Material) -> NodeRef<Material> {
        let name = name.unwrap_or_else(|| format!("mat_{}", self.materials.len()));
        let node_ref = self.materials.new_ref(Some(name));
        self.materials.insert(node_ref.clone(), material);
        node_ref
    }
    pub fn add_instance(&mut self, name: Option<String>, instance: Instance) -> NodeRef<Instance> {
        let name = name.unwrap_or_else(|| format!("instance_{}", self.instances.len()));
        let node_ref = self.instances.new_ref(Some(name));
        self.instances.insert(node_ref.clone(), instance);
        node_ref
    }
    pub fn write_to_file(
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
    pub unsafe fn compact(&mut self, name: String) -> std::io::Result<()> {
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
        let buffer = self.add_buffer(Some(name), buffer);
        // now update all buffer views
        self.buffer_views
            .par_iter_mut()
            .for_each(|(_, buffer_view)| {
                let offset = buf_ref_to_offset[&buffer_view.buffer];
                buffer_view.offset += offset;
                buffer_view.buffer = buffer.clone();
            });
        Ok(())
    }
}
/// Represents a loaded scene.
pub trait SceneView {
    fn buffer_as_slice(&self, buffer: &NodeRef<Buffer>) -> &[u8];
    fn buffer_view_as_slice(&self, buffer_view: &NodeRef<BufferView>) -> &[u8];
    fn scene(&self) -> &Scene;
}

/// A scene that is backed by memory.
/// It requires all buffers to be embedded/slice
pub struct MemoryScene {
    pub scene: Scene,
}
impl MemoryScene {
    pub unsafe fn new(mut scene: Scene) -> std::io::Result<Self> {
        scene.embed()?;
        scene.buffers.par_iter_mut().for_each(|(_, buffer)| {
            buffer.into_binary_inplace();
        });
        Ok(Self { scene })
    }
}
impl SceneView for MemoryScene {
    fn buffer_as_slice(&self, buffer: &NodeRef<Buffer>) -> &[u8] {
        let buffer = &self.scene.buffers[buffer];
        unsafe { buffer.as_binary_data() }
    }
    fn buffer_view_as_slice(&self, buffer_view: &NodeRef<BufferView>) -> &[u8] {
        let buffer = &self.scene.buffer_views[buffer_view].buffer;
        match &self.scene.buffers[buffer] {
            Buffer::Slice { slice } => unsafe { slice.as_slice() },
            Buffer::EmbeddedBinary { data } => data.as_slice(),
            _ => panic!(),
        }
    }
    fn scene(&self) -> &Scene {
        &self.scene
    }
}
/// A scene that is backed by memory mapped files.
pub struct MmapScene {
    pub scene: Scene,
    pub path_to_mmap: HashMap<PathBuf, Arc<Mmap>>,
}
impl MmapScene {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let abs_path = std::fs::canonicalize(path)?;
        let parent = abs_path.parent().unwrap();
        let scene_json = std::fs::read_to_string(&abs_path)?;
        let mut scene: Scene = serde_json::from_str(&scene_json)?;
        let mut path_to_mmap = HashMap::<PathBuf, Arc<Mmap>>::new();
        with_current_dir(parent, || -> std::io::Result<Self> {
            for (_, b) in scene.buffers.inner_mut() {
                match b {
                    Buffer::Path { path, length } => {
                        let path = std::fs::canonicalize(path)?;
                        let mmap = if let Some(mmap) = path_to_mmap.get(&path) {
                            mmap.clone()
                        } else {
                            let file = std::fs::File::open(&path)?;
                            let mmap = unsafe { Arc::new(Mmap::map(&file)?) };
                            path_to_mmap.insert(path.clone(), mmap.clone());
                            mmap
                        };
                        if *length != mmap.len() as u64 {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!(
                                    "buffer size mismatch: expected {}, got {}",
                                    length,
                                    mmap.len()
                                ),
                            ));
                        }
                        *b = Buffer::Slice {
                            slice: ExtSlice::new(mmap.as_ptr() as u64, mmap.len() as u64),
                        };
                    }
                    _ => {}
                }

                b.into_binary_inplace();
            }
            Ok(Self {
                scene,
                path_to_mmap,
            })
        })
    }
}
impl SceneView for MmapScene {
    fn buffer_as_slice(&self, buffer: &NodeRef<Buffer>) -> &[u8] {
        match &self.scene.buffers[buffer] {
            Buffer::Slice { slice } => unsafe { slice.as_slice() },
            Buffer::EmbeddedBinary { data } => data.as_slice(),
            _ => panic!("buffer is not mapped"),
        }
    }
    fn buffer_view_as_slice(&self, buffer_view: &NodeRef<BufferView>) -> &[u8] {
        let buffer = &self.scene.buffer_views[buffer_view].buffer;
        let offset = self.scene.buffer_views[buffer_view].offset;
        let length = self.scene.buffer_views[buffer_view].length;
        let buffer = self.buffer_as_slice(buffer);
        &buffer[offset..offset + length]
    }
    fn scene(&self) -> &Scene {
        &self.scene
    }
}

fn with_current_dir<T>(path: impl AsRef<Path>, f: impl FnOnce() -> T) -> T {
    let path = path.as_ref();
    let old_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(path).unwrap();
    let ret = f();
    std::env::set_current_dir(old_dir).unwrap();
    ret
}
