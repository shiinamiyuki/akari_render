use crate::bsdf::*;
use crate::camera::*;
#[cfg(feature = "gpu")]
use crate::gpu::pt::WavefrontPathTracer;
// use crate::film::*;
use crate::integrator::ao::RTAO;
use crate::integrator::nrc::CachedPathTracer;
use crate::integrator::path::PathTracer;
use crate::integrator::*;
use crate::light::*;
use crate::ltc::GgxLtcBsdf;
// use crate::sampler::*;
use crate::scene::*;
use crate::scenegraph::*;
use crate::shape::*;
use crate::texture::ConstantTexture;
use crate::texture::ImageTexture;
use crate::texture::Texture;
use crate::util::FileResolver;
use crate::util::LocalFileResolver;
use crate::*;
use core::panic;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;

use serde_json::Value;

struct SceneLoaderContext<'a> {
    parent_path: &'a Path,
    graph: &'a node::Scene,
    shapes: Vec<Arc<dyn Shape>>,
    camera: Option<Arc<dyn Camera>>,
    lights: Vec<Arc<dyn Light>>,
    named_bsdfs: HashMap<String, Arc<dyn Bsdf>>,
    texture_power: HashMap<usize, Float>,
    mesh_cache: HashMap<String, Arc<TriangleMesh>>,
    file_resolver: Box<dyn FileResolver>,
    gpu: bool,
}
impl<'a> SceneLoaderContext<'a> {
    fn load_texture(&self, node: &node::Texture) -> Arc<dyn Texture> {
        match node {
            node::Texture::Float(f) => Arc::new(ConstantTexture { value: *f }),
            node::Texture::Float3(f3) => Arc::new(ConstantTexture {
                value: Spectrum::from_rgb_linear(f3),
            }),
            node::Texture::Srgb(srgb) => Arc::new(ConstantTexture {
                value: Spectrum::from_srgb(&srgb.cast::<Float>()),
            }),
            node::Texture::SrgbU8(srgb) => Arc::new(ConstantTexture {
                value: Spectrum::from_srgb(&(srgb.cast::<Float>() / 255.0)),
            }),
            node::Texture::Hsv(hsv) => {
                let rgb = hsv_to_rgb(&hsv.cast::<Float>());
                Arc::new(ConstantTexture {
                    value: Spectrum::from_srgb(&rgb),
                })
            }
            node::Texture::Hex(_) => todo!(),
            node::Texture::Image(path) => {
                let file = self.resolve_file(path);
                let reader = BufReader::new(file);
                let reader = image::io::Reader::new(reader)
                    .with_guessed_format()
                    .unwrap();
                let img = reader.decode().unwrap().into_rgb8();
                Arc::new(ImageTexture::<Spectrum>::from_rgb_image(&img))
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
    fn power(&mut self, tex: Arc<dyn Texture>) -> Float {
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
                            let bson_data = bson::Document::from_reader(&mut file).unwrap();
                            bson::from_document::<TriangleMesh>(bson_data).unwrap()
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
                position: pos.cast::<Float>(),
                emission: self.load_texture(emission),
            }),
        }
    }
    fn load_transform(&self, trs: node::TRS) -> Transform {
        let mut m = glm::identity();
        let node::TRS {
            translate: t,
            rotate: r,
            scale: s,
        } = trs;
        let (t, r, s) = (t.cast::<Float>(), r.cast::<Float>(), s.cast::<Float>());
        let r: Vec3 = na::SVector::from_iterator(r.iter().map(|x| x.to_radians()));
        m = glm::scale(&glm::identity(), &s) * m;
        m = glm::rotate(&glm::identity(), r[0], &vec3(1.0, 0.0, 0.0)) * m;
        m = glm::rotate(&glm::identity(), r[1], &vec3(0.0, 1.0, 0.0)) * m;
        m = glm::rotate(&glm::identity(), r[2], &vec3(0.0, 0.0, 1.0)) * m;
        m = glm::translate(&glm::identity(), &t) * m;

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
                &vec2(res.0, res.1),
                &self.load_transform(transform),
                fov.to_radians() as Float,
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

pub fn load_scene<R: FileResolver>(path: &Path, gpu_mode: bool, accel: &str) -> Scene {
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
        file_resolver: Box::new(LocalFileResolver {
            paths: vec![PathBuf::from(parent_path)],
        }),
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
        "bdpt" => {
            let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
            let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as usize;
            let debug = (|| json.get("bdpt_debug")?.as_bool())().unwrap_or(false);
            Box::new(bdpt::Bdpt {
                spp,
                max_depth,
                debug,
            })
        }
        "mmlt" => {
            let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
            let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as usize;
            let n_bootstrap = (|| json.get("n_bootstrap")?.as_u64())().unwrap_or(100000) as usize;
            let n_chains = (|| json.get("n_chains")?.as_u64())().unwrap_or(1024) as usize;
            let direct_spp = (|| json.get("direct_spp")?.as_u64())().unwrap_or(16) as u32;
            Box::new(mmlt::MMLT {
                spp,
                max_depth: max_depth as u32,
                n_bootstrap,
                n_chains,
                direct_spp,
            })
        }
        "sppm" => {
            let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as usize;
            let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as usize;
            let n_photons = (|| json.get("n_photons")?.as_u64())().unwrap_or(100000) as usize;
            let initial_radius = (|| json.get("initial_radius")?.as_f64())().unwrap_or(0.1) as f32;
            Box::new(sppm::Sppm {
                iterations: spp,
                max_depth,
                n_photons,
                initial_radius,
            })
        }
        "ao" | "rtao" => {
            let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
            Box::new(RTAO { spp })
        }
        "cached" | "nrc" => {
            let spp = (|| json.get("spp")?.as_u64())().unwrap_or(16) as u32;
            let max_depth = (|| json.get("max_depth")?.as_u64())().unwrap_or(3) as u32;
            let batch_size = (|| json.get("batch_size")?.as_u64())().unwrap_or(512) as u32;
            let training_iters = (|| json.get("training_iters")?.as_u64())().unwrap_or(1024) as u32;
            let visualize_cache = (|| json.get("visualize_cache")?.as_bool())().unwrap_or(false);
            let learning_rate = (|| json.get("learning_rate")?.as_f64())().unwrap_or(0.001) as f32;
            Box::new(CachedPathTracer {
                spp,
                max_depth,
                visualize_cache,
                batch_size,
                training_iters,
                learning_rate,
            })
        }
        _ => unimplemented!(),
    }
}
