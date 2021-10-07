use crate::*;
use serde::{Deserialize, Serialize};
pub mod node {
    use super::*;

    #[derive(Clone, Copy, Serialize, Deserialize)]
    pub struct TRS {
        pub translate: glm::Vec3,
        pub rotate: glm::Vec3,
        pub scale: glm::Vec3,
    }
    impl Default for TRS {
        fn default() -> Self {
            Self {
                translate: glm::zero(),
                rotate: glm::zero(),
                scale: glm::vec3(1.0, 1.0, 1.0),
            }
        }
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub enum Texture {
        Float3(glm::Vec3),
        Float(f32),
        Srgb(glm::Vec3),
        SrgbU8(glm::UVec3),
        Hsv(glm::Vec3),
        Hex(glm::Vec3),
        Image(String),
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub enum Bsdf {
        Diffuse {
            color: Texture,
        },
        Principled {
            color: Texture,
            subsurface: Texture,
            subsurface_radius: Texture,
            subsurface_color: Texture,
            metallic: Texture,
            specular: Texture,
            specular_tint: Texture,
            roughness: Texture,
            anisotropic: Texture,
            anisotropic_rotation: Texture,
            sheen: Texture,
            sheen_tint: Texture,
            clearcoat: Texture,
            clearcoat_roughness: Texture,
            ior: Texture,
            transmission: Texture,
            emission: Texture,
            hint: String,
        },
        Named(String),
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub enum Shape {
        Mesh(String, Bsdf),
    }
    #[derive(Clone, Serialize, Deserialize)]
    pub enum Light {
        Point { pos: glm::Vec3, emission: Texture },
    }
    #[derive(Clone, Serialize, Deserialize)]
    pub enum Camera {
        Perspective {
            res: (u32, u32),
            fov: f32, // in degress
            lens_radius: f32,
            focal: f32,
            transform: TRS,
        },
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub struct Scene {
        pub named_bsdfs: HashMap<String, Bsdf>,
        pub camera: Camera,
        pub lights: Vec<Light>,
        pub shapes: Vec<Shape>,
    }
}
