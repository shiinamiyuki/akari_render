use serde::{Deserialize, Serialize};

pub mod node {
    use std::collections::HashMap;

    use super::*;
    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum CoordinateSystem {
        Akari,   //RightHandYUp
        Blender, //RightHandZUp
    }
    impl Default for CoordinateSystem {
        fn default() -> Self {
            Self::Akari
        }
    }
    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct TRS {
        pub translate: [f32; 3],
        pub rotate: [f32; 3],
        pub scale: [f32; 3],
        #[serde(default)]
        pub coord_sys: CoordinateSystem,
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
                coord_sys: Default::default(),
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
        Float(Texture),
        #[serde(rename = "spectrum")]
        Spectrum(Texture),
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
    #[serde(tag = "type")]
    pub enum Texture {
        Float {
            value: f32,
        },
        #[serde(rename = "linear")]
        SRgbLinear {
            values: [f32; 3],
        },
        #[serde(rename = "srgb")]
        SRgb {
            values: [f32; 3],
        },
        #[serde(rename = "srgb8")]
        SRgbU8 {
            values: [u8; 3],
        },
        #[serde(rename = "image")]
        Image {
            path: String,
            #[serde(default = "default_colorspace")]
            colorspace: String,
        },
    }
    fn default_ior() -> f32 {
        1.502
    }
    fn default_dispersion() -> f32 {
        0.0
    }
    #[derive(Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum Bsdf {
        #[serde(rename = "diffuse")]
        Diffuse { color: Texture },
        #[serde(rename = "glass")]
        Glass {
            #[serde(default = "default_ior")]
            ior: f32,
            #[serde(default = "default_dispersion")]
            dispersion: f32,
            kr: Texture,
            kt: Texture,
            roughness: Texture,
        },
        #[serde(rename = "principled")]
        Principled {
            color: Texture,
            subsurface: Texture,
            subsurface_radius: Texture,
            subsurface_color: Texture,
            subsurface_ior: Texture,
            metallic: Texture,
            specular_tint: Texture,
            roughness: Texture,
            anisotropic: Texture,
            anisotropic_rotation: Texture,
            sheen: Texture,
            sheen_tint: Texture,
            clearcoat: Texture,
            clearcoat_roughness: Texture,
            ior: f32,
            transmission: Texture,
            emission: Texture,
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
        Point { pos: [f32; 3], emission: Texture },
        #[serde(rename = "spot")]
        Spot {
            transform: Transform,
            emission: Texture,
            falloff: f32,
            max_angle: f32,
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

    pub enum GenericTextureRefMut<'a> {
        Float(&'a mut Texture),
        Spectrum(&'a mut Texture),
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
