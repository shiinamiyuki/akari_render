use crate::*;
use serde::{Deserialize, Serialize};
pub mod node {
    use std::collections::HashMap;

    use super::*;

    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct TRS {
        pub translate: [f32; 3],
        pub rotate: [f32; 3],
        pub scale: [f32; 3],
    }
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct LookAt {
        pub eye: [f32; 3],
        pub center: [f32; 3],
        pub up: [f32; 3],
    }
    #[derive(Clone, Copy, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum Transform {
        LookAt(LookAt),
        TRS(TRS),
    }
    impl Default for TRS {
        fn default() -> Self {
            Self {
                translate: [0.0; 3],
                rotate: [0.0; 3],
                scale: [1.0, 1.0, 1.0],
            }
        }
    }
    fn default_colorspace() -> String {
        "srgb".into()
    }
    #[derive(Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum ShaderGraphNode {
        #[serde(rename = "float")]
        Float(FloatTexture),
        #[serde(rename = "spectrum")]
        Spectrum(SpectrumTexture),
        #[serde(rename = "mix")]
        Mix {
            frac: String,
            tex_a: String,
            tex_b: String,
        },
        #[serde(rename = "noise")]
        Noise { pattern: String, dimension: u8 },
    }
    #[derive(Clone, Serialize, Deserialize)]
    pub struct ShaderGraph {
        pub nodes: HashMap<String, ShaderGraphNode>,
        pub resolution: usize,
        pub precompute: bool,
        pub cache: Option<String>,
    }
    #[derive(Clone, Serialize, Deserialize)]
    pub struct TextureCache {
        pub path: String,
    }
    #[derive(Clone, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum FloatTexture {
        Float(f32),
        Image(String),
        CachedImage {
            path: String,
            #[serde(default)]
            cache: Option<TextureCache>,
        },
    }
    #[derive(Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum SpectrumTexture {
        #[serde(rename = "linear")]
        SRgbLinear { values: [f32; 3] },
        #[serde(rename = "srgb")]
        SRgb { values: [f32; 3] },
        #[serde(rename = "srgb8")]
        SRgbU8 { values: [u8; 3] },
        #[serde(rename = "image")]
        Image {
            path: String,
            #[serde(default = "default_colorspace")]
            colorspace: String,
            #[serde(default)]
            cache: Option<TextureCache>,
        },
    }
    #[derive(Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum Bsdf {
        #[serde(rename = "diffuse")]
        Diffuse { color: SpectrumTexture },
        #[serde(rename = "glass")]
        Glass {
            kr: SpectrumTexture,
            kt: SpectrumTexture,
        },
        #[serde(rename = "principled")]
        Principled {
            color: SpectrumTexture,
            subsurface: FloatTexture,
            subsurface_radius: SpectrumTexture,
            subsurface_color: SpectrumTexture,
            subsurface_ior: FloatTexture,
            metallic: FloatTexture,
            specular: FloatTexture,
            specular_tint: FloatTexture,
            roughness: FloatTexture,
            anisotropic: FloatTexture,
            anisotropic_rotation: FloatTexture,
            sheen: FloatTexture,
            sheen_tint: FloatTexture,
            clearcoat: FloatTexture,
            clearcoat_roughness: FloatTexture,
            ior: FloatTexture,
            transmission: FloatTexture,
            emission: SpectrumTexture,
        },
    }

    #[derive(Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum Shape {
        #[serde(rename = "mesh")]
        Mesh {
            path: String,
            bsdf: String,
            #[serde(default)]
            transform: Option<Transform>,
        },
    }
    #[derive(Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum Light {
        #[serde(rename = "point")]
        Point {
            pos: [f32; 3],
            emission: SpectrumTexture,
        },
    }
    #[derive(Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum Camera {
        #[serde(rename = "perspective")]
        Perspective {
            res: (u32, u32),
            fov: f32, // in degress
            lens_radius: f32,
            focal: f32,
            transform: Transform,
        },
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub struct Scene {
        pub bsdfs: HashMap<String, Bsdf>,
        pub camera: Camera,
        pub lights: Vec<Light>,
        pub shapes: Vec<Shape>,
        // #[serde(default = "Vec::new")]
        // pub shaders: Vec<ShaderGraph>,
    }
}

pub mod api {
    use serde::{Deserialize, Serialize};

    pub type Handle = String;
    #[derive(Clone, Serialize, Deserialize, Debug)]
    pub struct ImportMesh {
        pub filename: String,
        pub scene: Handle,
    }
    #[derive(Clone, Serialize, Deserialize, Debug)]
    pub struct UpdateMesh {
        pub filename: String,
        pub mesh: Handle,
        pub scene: Handle,
    }
    #[derive(Clone, Serialize, Deserialize, Debug)]
    pub struct ConvertTextureTiled {
        pub filename: String,
        pub scene: Handle,
        pub tile_size: usize,
    }
    pub enum Command {}
}
