use crate::*;
use serde::{Deserialize, Serialize};

pub mod node {
    use std::collections::HashMap;

    use akari_common::ordered_float::Float;

    use super::*;
    #[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum CoordinateSystem {
        // RightHandYUp,
        // RightHandZUp,
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
        Diffuse { color: SpectrumTexture },
        #[serde(rename = "glass")]
        Glass {
            #[serde(default = "default_ior")]
            ior: f32,
            #[serde(default = "default_dispersion")]
            dispersion: f32,
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
        #[serde(rename = "spot")]
        Spot {
            transform: Transform,
            emission: SpectrumTexture,
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
        Float(&'a mut FloatTexture),
        Spectrum(&'a mut SpectrumTexture),
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
    impl Bsdf {
        pub fn foreach_texture<F: FnMut(GenericTextureRefMut<'_>)>(&mut self, mut f: F) {
            match self {
                Bsdf::Diffuse { color } => f(GenericTextureRefMut::Spectrum(color)),
                Bsdf::Glass { kr, kt, .. } => {
                    f(GenericTextureRefMut::Spectrum(kr));
                    f(GenericTextureRefMut::Spectrum(kt));
                }
                Bsdf::Principled {
                    color,
                    subsurface,
                    subsurface_radius,
                    subsurface_color,
                    subsurface_ior,
                    metallic,
                    specular,
                    specular_tint,
                    roughness,
                    anisotropic,
                    anisotropic_rotation,
                    sheen,
                    sheen_tint,
                    clearcoat,
                    clearcoat_roughness,
                    ior,
                    transmission,
                    emission,
                } => {
                    f(GenericTextureRefMut::Spectrum(color));
                    f(GenericTextureRefMut::Float(subsurface));
                    f(GenericTextureRefMut::Spectrum(subsurface_radius));
                    f(GenericTextureRefMut::Spectrum(subsurface_color));
                    f(GenericTextureRefMut::Float(subsurface_ior));
                    f(GenericTextureRefMut::Float(metallic));
                    f(GenericTextureRefMut::Float(specular));
                    f(GenericTextureRefMut::Float(specular_tint));
                    f(GenericTextureRefMut::Float(roughness));
                    f(GenericTextureRefMut::Float(anisotropic));
                    f(GenericTextureRefMut::Float(anisotropic_rotation));
                    f(GenericTextureRefMut::Float(sheen));
                    f(GenericTextureRefMut::Float(sheen_tint));
                    f(GenericTextureRefMut::Float(clearcoat));
                    f(GenericTextureRefMut::Float(clearcoat_roughness));
                    f(GenericTextureRefMut::Float(ior));
                    f(GenericTextureRefMut::Float(transmission));
                    f(GenericTextureRefMut::Spectrum(emission));
                }
            }
        }
    }
    impl Scene {
        pub fn foreach_ext_files<F: FnMut(&mut String)>(&mut self, mut f: F) {
            for shape in &mut self.shapes {
                match shape {
                    Shape::Mesh { path, .. } => f(path),
                }
            }
            for (_, bsdf) in &mut self.bsdfs {
                bsdf.foreach_texture(|tex| match tex {
                    GenericTextureRefMut::Float(tex) => match tex {
                        FloatTexture::Image(img) => f(img),
                        _ => {}
                    },
                    GenericTextureRefMut::Spectrum(tex) => match tex {
                        SpectrumTexture::Image { path, .. } => f(path),
                        _ => {}
                    },
                });
            }
        }
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
