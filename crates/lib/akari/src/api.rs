use crate::bsdf::*;
use crate::camera::*;
#[cfg(feature = "gpu")]
use crate::gpu::pt::WavefrontPathTracer;
// use crate::film::*;
// use crate::integrator::ao::RTAO;
// use crate::integrator::normalvis::NormalVis;
// use crate::integrator::nrc::CachedPathTracer;
use crate::integrator::path::PathTracer;
// use crate::integrator::spath::StreamPathTracer;

use crate::bsdf::ltc::GgxLtcBsdf;
use crate::light::*;
// use crate::sampler::*;
use crate::scene::*;
use crate::scenegraph::*;
use crate::shape::*;
use crate::texture::{ConstantFloatTexture, ConstantRgbTexture};
// use crate::texture::ImageTexture;
use crate::texture::Texture;
use crate::util::binserde::Decode;
use crate::util::FileResolver;
use crate::util::LocalFileResolver;
use crate::*;
use core::panic;
use glam::*;
// use integrator::bdpt;
// use integrator::erpt;
// use integrator::mmlt;
// use integrator::pssmlt;
use integrator::Integrator;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

struct SceneLoaderContext<'a> {
    parent_path: &'a Path,
    graph: &'a node::Scene,
    shapes: Vec<Arc<dyn Shape>>,
    camera: Option<Arc<dyn Camera>>,
    lights: Vec<Arc<dyn Light>>,
    named_bsdfs: HashMap<String, Arc<dyn Bsdf>>,
    texture_power: HashMap<usize, f32>,
    mesh_cache: HashMap<String, Arc<TriangleMesh>>,
    file_resolver: Arc<dyn FileResolver + Send + Sync>,
    gpu: bool,
    ooc: OocOptions,
}
impl<'a> SceneLoaderContext<'a> {
    fn load_texture(&mut self, node: &node::Texture) -> Arc<dyn Texture> {
        let colorspace = RgbColorSpace::new(RgbColorSpaceId::SRgb);
        match node {
            node::Texture::Float(f) => Arc::new(ConstantFloatTexture(*f)),
            node::Texture::Float3(f3) => {
                Arc::new(ConstantRgbTexture::new(Vec3::from(*f3), colorspace))
            }
            node::Texture::Srgb(srgb) => Arc::new(ConstantRgbTexture::new(
                srgb_to_linear(Vec3::from(*srgb)),
                colorspace,
            )),
            node::Texture::SrgbU8(srgb) => Arc::new(ConstantRgbTexture::new(
                srgb_to_linear(
                    UVec3::from([srgb[0] as u32, srgb[1] as u32, srgb[2] as u32]).as_vec3() / 255.0,
                ),
                colorspace,
            )),
            node::Texture::Hsv(hsv) => {
                let rgb = hsv_to_rgb(Vec3::from(*hsv));
                Arc::new(ConstantRgbTexture::new(srgb_to_linear(rgb), colorspace))
            }
            node::Texture::Hex(_) => todo!(),
            node::Texture::Image(path) => {
                todo!()
                // let file = self.resolve_file(path);
                // let reader = BufReader::new(file);
                // let reader = image::io::Reader::new(reader)
                //     .with_guessed_format()
                //     .unwrap();
                // let img = reader.decode().unwrap().into_rgb8();
                // if cfg!(feature = "gpu") || !self.ooc.enable_ooc {
                //     Arc::new(ImageTexture::<Spectrum>::from_rgb_image(&img, false))
                // } else {
                //     Arc::new(ImageTexture::<Spectrum>::from_rgb_image(&img, true))
                // }
            }
        }
    }
    fn load_named_bsdf(&mut self, name: &String) -> Arc<dyn Bsdf> {
        if let Some(bsdf) = self.named_bsdfs.get(name) {
            return bsdf.clone();
        } else {
            if let Some(node) = self.graph.named_bsdfs.get(name) {
                let bsdf = self.load_bsdf(node);
                self.named_bsdfs.insert(name.clone(), bsdf.clone());
                return bsdf;
            } else {
                println!("no bsdf named {}", name);
                std::process::exit(-1);
            }
        }
    }
    #[allow(dead_code)]
    fn power(&mut self, tex: Arc<dyn Texture>) -> f32 {
        let addr = Arc::into_raw(tex.clone()).cast::<()>() as usize;
        if let Some(p) = self.texture_power.get(&addr) {
            return *p;
        }
        let p = tex.power();
        self.texture_power.insert(addr, p);
        p
    }
    fn load_bsdf(&mut self, node: &node::Bsdf) -> Arc<dyn Bsdf> {
        match node {
            node::Bsdf::Diffuse { color } => Arc::new(DiffuseBsdf {
                color: self.load_texture(color),
            }),
            node::Bsdf::Principled {
                color,
                subsurface,
                subsurface_color,
                subsurface_radius,
                sheen,
                sheen_tint,
                specular,
                specular_tint,
                metallic,
                roughness,
                anisotropic,
                anisotropic_rotation,
                clearcoat,
                clearcoat_roughness,
                ior,
                transmission,
                emission,
                hint,
            } => {
                if !self.gpu {
                    if hint == "ltc" {
                        let color = self.load_texture(color);
                        let emission = self.load_texture(emission);

                        let bsdf = Arc::new(MixBsdf {
                            frac: self.load_texture(metallic),
                            bsdf_a: DiffuseBsdf {
                                color: color.clone(),
                            },
                            bsdf_b: GgxLtcBsdf {
                                roughness: self.load_texture(roughness),
                                color: color.clone(),
                            },
                        });
                        let bsdf: Arc<dyn Bsdf> = if emission.power() > 0.0 {
                            Arc::new(EmissiveBsdf {
                                base: bsdf,
                                emission,
                            })
                        } else {
                            bsdf
                        };
                        bsdf
                    } else {
                        unimplemented!("currently only ltc bsdf is supported");
                    }
                } else {
                    Arc::new(GPUBsdfProxy {
                        color: self.load_texture(color),
                        metallic: self.load_texture(metallic),
                        roughness: self.load_texture(roughness),
                        emission: self.load_texture(emission),
                    })
                }
            }
            node::Bsdf::Named(name) => self.load_named_bsdf(name),
        }
    }
    fn load_shape(&mut self, node: &node::Shape) -> Arc<dyn Shape> {
        match node {
            node::Shape::Mesh(path, bsdf_node) => {
                let mesh = {
                    if let Some(cache) = self.mesh_cache.get(path) {
                        cache.clone()
                    } else {
                        let mut file = self.resolve_file(path);
                        let model = Arc::new({
                            // let bson_data = bson::Document::from_reader(&mut file).unwrap();
                            // bson::from_document::<TriangleMesh>(bson_data).unwrap()
                            TriangleMesh::decode(&mut file).unwrap()
                        });
                        self.mesh_cache.insert(path.clone(), model.clone());
                        model
                    }
                };

                let bsdf = self.load_bsdf(bsdf_node);
                Arc::new(MeshInstanceProxy { mesh, bsdf })
            }
        }
    }
    fn load_light(&mut self, node: &node::Light) -> Arc<dyn Light> {
        match node {
            node::Light::Point { pos, emission } => Arc::new(PointLight {
                position: Vec3::from(*pos),
                emission: self.load_texture(emission),
            }),
        }
    }
    fn load_transform(&self, trs: node::TRS) -> Transform {
        let mut m = Mat4::IDENTITY;
        let node::TRS {
            translate: t,
            rotate: r,
            scale: s,
        } = trs;
        let (t, r, s) = (t.into(), r.into(), s.into());
        let r: Vec3 = r;
        let r = vec3(r.x.to_radians(), r.y.to_radians(), r.z.to_radians());
        m = Mat4::from_scale(s) * m;
        m = Mat4::from_axis_angle(vec3(1.0, 0.0, 0.0), r[0]) * m;
        m = Mat4::from_axis_angle(vec3(0.0, 1.0, 0.0), r[1]) * m;
        m = Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), r[2]) * m;
        m = Mat4::from_translation(t) * m;

        Transform::from_matrix(&m)
    }
    fn load_transform2(&self, trs: node::TR) -> Transform {
        let mut m = Mat4::IDENTITY;
        let node::TR {
            translate: t,
            rotate: r,
        } = trs;
        let (t, r) = (t.into(), r.into());
        let r: Vec3 = r;
        let r = vec3(r.x.to_radians(), r.y.to_radians(), r.z.to_radians());
        m = Mat4::from_axis_angle(vec3(1.0, 0.0, 0.0), r[0]) * m;
        m = Mat4::from_axis_angle(vec3(0.0, 1.0, 0.0), r[1]) * m;
        m = Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), r[2]) * m;
        m = Mat4::from_translation(t) * m;

        Transform::from_matrix(&m)
    }
    fn load(&mut self) {
        self.camera = Some(match self.graph.camera {
            node::Camera::Perspective {
                res,
                fov,
                lens_radius,
                focal,
                transform,
            } => Arc::new(PerspectiveCamera::new(
                uvec2(res.0, res.1),
                &self.load_transform2(transform),
                fov.to_radians() as f32,
            )),
        });
        for node in self.graph.shapes.iter() {
            let shape = self.load_shape(node);
            self.shapes.push(shape);
        }
        for light in self.graph.lights.iter() {
            let light = self.load_light(light);
            self.lights.push(light);
        }
    }
    fn resolve_file(&self, path: &String) -> File {
        let path = if cfg!(target_os = "windows") {
            path.replace("/", "\\")
        } else {
            path.replace("\\", "/")
        };
        if let Some(file) = self.file_resolver.resolve(path.as_ref()) {
            return file;
        }
        panic!("cannot resolve path {}", path);
    }
}
#[derive(Clone, Copy)]
pub struct OocOptions {
    pub enable_ooc: bool,
}
pub fn load_scene<R: FileResolver + Send + Sync>(
    path: &Path,
    gpu_mode: bool,
    accel: &str,
    ooc: OocOptions,
) -> Scene {
    let serialized = std::fs::read_to_string(path).unwrap();
    let canonical = std::fs::canonicalize(path).unwrap();
    let graph: node::Scene = serde_json::from_str(&serialized).unwrap();
    let parent_path = canonical.parent().unwrap();
    let mut ctx = SceneLoaderContext {
        parent_path,
        graph: &graph,
        shapes: vec![],
        lights: vec![],
        camera: None,
        ooc,
        file_resolver: Arc::new(LocalFileResolver::new(vec![PathBuf::from(parent_path)])),
        named_bsdfs: HashMap::new(),
        texture_power: HashMap::new(),
        mesh_cache: HashMap::new(),
        gpu: gpu_mode,
    };
    ctx.load();
    let scene = Scene::new(
        ctx.camera.unwrap(),
        ctx.shapes.clone(),
        ctx.mesh_cache
            .iter()
            .map(|(_, cache)| cache.clone())
            .collect(),
        ctx.lights.clone(),
        accel,
        gpu_mode,
    );

    println!("{} lights", scene.lights.len());

    scene
}
#[cfg(feature = "gpu")]
pub fn load_gpu_integrator(path: &Path) -> WavefrontPathTracer {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let json: Value = serde_json::from_reader(reader).unwrap();
    let ty = json.get("type").unwrap().as_str().unwrap();
    match ty {
        "pt" | "path" => {
            let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
            let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as u32;
            WavefrontPathTracer {
                spp,
                max_depth,
                training_iters: 0,
            }
        }
        "cached" | "nrc" => {
            {
                eprintln!(" \"gpu_nrc\" is disabled, will be back in the future");
                std::process::exit(1);
            }
            // let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
            // let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as u32;
            // // let batch_size = (|| json.get("batch_size")?.as_u64())().unwrap_or(512) as u32;
            // let training_iters = (|| json.get("training_iters")?.as_u64())().unwrap_or(1024) as u32;
            // // let visualize_cache = (|| json.get("visualize_cache")?.as_bool())().unwrap_or(false);
            // // let learning_rate = (|| json.get("learning_rate")?.as_f64())().unwrap_or(0.001) as f32;
            // WavefrontPathTracer {
            //     spp,
            //     max_depth,
            //     training_iters,
            // }
        }
        _ => {
            eprintln!("integrator {} is not supported on gpu", ty);
            std::process::exit(1);
        }
    }
}
pub fn load_integrator(path: &Path) -> Box<dyn Integrator> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let json: Value = serde_json::from_reader(reader).unwrap();
    let ty = json.get("type").unwrap().as_str().unwrap();
    match ty {
        "pt" | "path" => {
            let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
            let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as u32;
            Box::new(PathTracer { spp, max_depth })
        }
        // "spath" => {
        //     let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
        //     let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as u32;
        //     let batch_size = (|| json.get("batch_size")?.as_u64())().unwrap_or(1 << 15) as usize;
        //     let sort_rays = (|| json.get("sort_rays")?.as_bool())().unwrap_or(true);
        //     Box::new(StreamPathTracer {
        //         spp,
        //         max_depth,
        //         batch_size,
        //         sort_rays,
        //     })
        // }
        // "bdpt" => {
        //     let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
        //     let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as usize;
        //     let debug = (|| json.get("bdpt_debug")?.as_bool())().unwrap_or(false);
        //     Box::new(bdpt::Bdpt {
        //         spp,
        //         max_depth,
        //         debug,
        //     })
        // }
        // "pssmlt" => {
        //     let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
        //     let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as usize;
        //     let n_bootstrap = (|| json.get("n_bootstrap")?.as_u64())().unwrap_or(100000) as usize;
        //     let n_chains = (|| json.get("n_chains")?.as_u64())().unwrap_or(1024) as usize;
        //     let direct_spp = (|| json.get("direct_spp")?.as_u64())().unwrap_or(16) as u32;
        //     Box::new(pssmlt::Pssmlt {
        //         spp,
        //         max_depth: max_depth as u32,
        //         n_bootstrap,
        //         n_chains,
        //         direct_spp,
        //     })
        // }
        // "erpt" => {
        //     let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
        //     let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as usize;
        //     let n_bootstrap = (|| json.get("n_bootstrap")?.as_u64())().unwrap_or(100000) as usize;
        //     let mutations_per_chain =
        //         (|| json.get("mutations_per_chain")?.as_u64())().unwrap_or(100) as usize;
        //     let direct_spp = (|| json.get("direct_spp")?.as_u64())().unwrap_or(16) as u32;
        //     Box::new(erpt::Erpt {
        //         spp,
        //         max_depth: max_depth as u32,
        //         n_bootstrap,
        //         mutations_per_chain,
        //         direct_spp,
        //     })
        // }
        // "mmlt" => {
        //     let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
        //     let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as usize;
        //     let n_bootstrap = (|| json.get("n_bootstrap")?.as_u64())().unwrap_or(100000) as usize;
        //     let n_chains = (|| json.get("n_chains")?.as_u64())().unwrap_or(1024) as usize;
        //     let direct_spp = (|| json.get("direct_spp")?.as_u64())().unwrap_or(16) as u32;
        //     Box::new(mmlt::Mmlt {
        //         spp,
        //         max_depth: max_depth as u32,
        //         n_bootstrap,
        //         n_chains,
        //         direct_spp,
        //     })
        // }
        // "sppm" => {
        //     let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as usize;
        //     let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as usize;
        //     let n_photons = (|| json.get("n_photons")?.as_u64())().unwrap_or(100000) as usize;
        //     let initial_radius = (|| json.get("initial_radius")?.as_f64())().unwrap_or(0.1) as f32;
        //     Box::new(sppm::Sppm {
        //         iterations: spp,
        //         max_depth,
        //         n_photons,
        //         initial_radius,
        //     })
        // }
        // "ao" | "rtao" => {
        //     let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
        //     Box::new(RTAO { spp })
        // }
        // "normal" => Box::new(NormalVis {}),
        // "cached" | "nrc" => {
        //     let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
        //     let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as u32;
        //     let batch_size = (|| json.get("batch_size")?.as_u64())().unwrap_or(512) as u32;
        //     let training_iters = (|| json.get("training_iters")?.as_u64())().unwrap_or(1024) as u32;
        //     let visualize_cache = (|| json.get("visualize_cache")?.as_bool())().unwrap_or(false);
        //     let learning_rate = (|| json.get("learning_rate")?.as_f64())().unwrap_or(0.001) as f32;
        //     Box::new(CachedPathTracer {
        //         spp,
        //         max_depth,
        //         visualize_cache,
        //         batch_size,
        //         training_iters,
        //         learning_rate,
        //     })
        // }
        _ => unimplemented!(),
    }
}
