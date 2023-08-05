use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::Arc;
use std::{path::PathBuf, rc::Rc};

use luisa::PixelStorage;

use crate::camera::PerspectiveCamera;
use crate::color::{glam_srgb_to_linear, Color, ColorRepr, FlatColor};
use crate::light::area::AreaLight;
use crate::light::{
    FlatLightSample, FlatLightSampleExpr, LightAggregate, LightEvalContext, LightEvaluator,
    WeightedLightDistribution,
};

use crate::scenegraph::node::CoordinateSystem;
use crate::surface::diffuse::DiffuseSurface;
use crate::surface::glass::GlassSurface;
use crate::surface::principled::PrincipledSurface;
use crate::surface::{
    BsdfEvalContext, BsdfEvaluator, BsdfSample, FlatBsdfEvalResult, FlatBsdfSample,
    BSDF_EVAL_COLOR, BSDF_EVAL_PDF,
};
use crate::texture::{
    ConstFloatTexture, ConstRgbTexture, ImageRgbTexture, TextureEvalContext, TextureEvaluator,
};
use crate::util::binserde::Decode;
use crate::util::{FileResolver, LocalFileResolver};
use crate::{
    camera::Camera,
    geometry::*,
    interaction::*,
    light::{Light, LightDistribution},
    mesh::*,
    scenegraph::node,
    surface::{Bsdf, Surface},
    texture::Texture,
    *,
};

pub struct Scene {
    pub textures: Polymorphic<PolyKey, dyn Texture>,
    pub surfaces: Polymorphic<PolyKey, dyn Surface>,
    pub lights: LightAggregate,
    pub meshes: Arc<MeshAggregate>,
    pub camera: Box<dyn Camera>,
    pub device: Device,
    pub image_textures: BindlessArray,
    pub env_map: Buffer<TagIndex>,
}
pub struct Evaluators {
    pub color_repr: ColorRepr,
    pub texture: Arc<TextureEvaluator>,
    pub bsdf: Arc<BsdfEvaluator>,
    pub light: Arc<LightEvaluator>,
}

impl Scene {
    pub fn evaluators(self: &Arc<Self>, color_repr: ColorRepr) -> Evaluators {
        let texture = self.texture_evaluator(color_repr);
        let light = self.light_evaluator(color_repr, texture.clone());
        let bsdf = self.bsdf_evaluator(color_repr, texture.clone());
        Evaluators {
            color_repr,
            texture,
            bsdf,
            light,
        }
    }
    pub fn light_evaluator(
        self: &Arc<Self>,
        color_repr: ColorRepr,
        texture_eval: Arc<TextureEvaluator>,
    ) -> Arc<LightEvaluator> {
        let le = {
            let texture_eval = texture_eval.clone();
            let scene = self.clone();

            self.device
                .create_callable::<(Expr<Ray>, Expr<SurfaceInteraction>), Expr<FlatColor>>(
                    &|ray, si: Expr<SurfaceInteraction>| {
                        let inst_id = si.inst_id();
                        let instance = scene.meshes.mesh_instances.var().read(inst_id);
                        if_!(instance.light().valid(), {
                            let ctx = LightEvalContext {
                                meshes: &scene.meshes,
                                texture: &texture_eval,
                                color_repr,
                            };
                            scene.lights.le(ray, si, &ctx).flatten()
                        }, else {
                          Color::zero(color_repr).flatten()
                        })
                    },
                )
        };
        let sample = {
            let texture_eval = texture_eval.clone();
            let scene = self.clone();
            self.device
                .create_callable::<(Expr<PointNormal>, Expr<Float3>), Expr<FlatLightSample>>(
                    &|pn, u| {
                        let ctx = LightEvalContext {
                            meshes: &scene.meshes,
                            texture: &texture_eval,
                            color_repr,
                        };
                        let sample = scene.lights.sample_direct(pn, u.x(), u.yz(), &ctx);
                        FlatLightSampleExpr::new(
                            sample.li.flatten(),
                            sample.pdf,
                            sample.wi,
                            sample.shadow_ray,
                            sample.n,
                        )
                    },
                )
        };
        let pdf = {
            let scene = self.clone();
            self.device
                .create_callable::<(Expr<SurfaceInteraction>, Expr<PointNormal>), Expr<f32>>(
                    &|si, pn| {
                        let inst_id = si.inst_id();
                        let instance = scene.meshes.mesh_instances.var().read(inst_id);
                        if_!(instance.light().valid(), {
                            let ctx = LightEvalContext {
                                meshes: &scene.meshes,
                                texture: &texture_eval,
                                color_repr: color_repr,
                            };
                            scene.lights.pdf_direct(si, pn, &ctx)
                        }, else {
                            const_(0.0f32)
                        })
                    },
                )
        };
        Arc::new(LightEvaluator {
            color_repr,
            le,
            sample,
            pdf,
        })
    }
    pub fn texture_evaluator(&self, color_repr: ColorRepr) -> Arc<TextureEvaluator> {
        let texture = self
            .device
            .create_callable::<(Expr<TagIndex>, Expr<SurfaceInteraction>), Expr<Float4>>(
                &|tex: Expr<TagIndex>, si: Expr<SurfaceInteraction>| {
                    let ctx = TextureEvalContext { scene: self };
                    self.textures
                        .get(tex)
                        .dispatch(|_, _, tex| tex.evaluate(si, &ctx))
                },
            );
        Arc::new(TextureEvaluator {
            color_repr,
            texture,
        })
    }
    pub fn bsdf_evaluator(
        self: &Arc<Self>,
        color_repr: ColorRepr,
        texture_eval: Arc<TextureEvaluator>,
    ) -> Arc<BsdfEvaluator> {
        let scene = self.clone();
        let bsdf = {
            let texture_eval = texture_eval.clone();
            self.device.create_callable::<(
                Expr<TagIndex>,
                Expr<SurfaceInteraction>,
                Expr<Float3>,
                Expr<Float3>,
                Expr<f32>,
                Expr<u32>,
            ), Expr<FlatBsdfEvalResult>>(
                &|surface: Expr<TagIndex>,
                  si: Expr<SurfaceInteraction>,
                  wo: Expr<Float3>,
                  wi: Expr<Float3>,
                  u_select: Expr<f32>,
                  mode: Expr<u32>| {
                    let ctx = BsdfEvalContext {
                        texture: &texture_eval,
                        color_repr: color_repr,
                    };
                    let (color, pdf) = scene.surfaces.get(surface).dispatch(|_, _, surface| {
                        let closure = surface.closure(si, &ctx);
                        let color = if_!((mode & BSDF_EVAL_COLOR).cmpne(0), {
                            closure.evaluate(wo, wi, &ctx)
                        }, else {
                            Color::zero(color_repr)
                        });
                        let pdf = if_!((mode & BSDF_EVAL_PDF).cmpne(0), {
                            closure.pdf(wo, wi, &ctx)
                        }, else {
                            0.0.into()
                        });
                        (color, pdf)
                    });
                    struct_!(FlatBsdfEvalResult {
                        color: color.flatten(),
                        pdf: pdf,
                        lobe_roughness: const_(0.0f32) //TODO
                    })
                },
            )
        };
        let albedo = {
            let scene = self.clone();
            let texture_eval = texture_eval.clone();
            self.device.create_callable::<(
                Expr<TagIndex>,
                Expr<SurfaceInteraction>,
                Expr<Float3>,
            ), Expr<FlatColor>>(&|surface: Expr<TagIndex>,
                      si: Expr<SurfaceInteraction>,
                      wo: Expr<Float3>| {
                    let ctx = BsdfEvalContext {
                        texture: &texture_eval,
                        color_repr: color_repr,
                    };
                    let color = scene.surfaces.get(surface).dispatch(|_, _, surface| {
                        let closure = surface.closure(si, &ctx);
                        closure.albedo(wo, &ctx)
                    });
                    color.flatten()
                },
            )
        };
        let scene = self.clone();
        let bsdf_sample = {
            let texture_eval = texture_eval.clone();
            self.device.create_callable::<(
                Expr<TagIndex>,
                Expr<SurfaceInteraction>,
                Expr<Float3>,
                Expr<Float3>,
            ), Expr<FlatBsdfSample>>(
                &|surface: Expr<TagIndex>,
                  si: Expr<SurfaceInteraction>,
                  wo: Expr<Float3>,
                  u: Expr<Float3>| {
                    let ctx = BsdfEvalContext {
                        texture: &texture_eval,
                        color_repr: color_repr,
                    };
                    let sample = scene.surfaces.get(surface).dispatch(|_, _, surface| {
                        let closure = surface.closure(si, &ctx);
                        closure.sample(wo, u.x(), u.yz(), &ctx)
                    });
                    struct_!(FlatBsdfSample {
                        wi: sample.wi,
                        pdf: sample.pdf,
                        valid: sample.valid,
                        color: sample.color.flatten(),
                        lobe_roughness: sample.lobe_roughness
                    })
                },
            )
        };
        Arc::new(BsdfEvaluator {
            color_repr,
            bsdf,
            bsdf_sample,
            albedo,
        })
    }
    pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Self> {
        let path = PathBuf::from(path.as_ref());
        let canonical = fs::canonicalize(&path).unwrap();
        let parent_path = canonical.parent().unwrap();
        let serialized = std::fs::read_to_string(path).unwrap();
        Self::load_from_str(
            device,
            &serialized,
            Arc::new(LocalFileResolver::new(vec![PathBuf::from(parent_path)])),
        )
    }
    pub fn load_from_str(
        device: Device,
        desc: &str,
        file_resolver: Arc<dyn FileResolver + Send + Sync>,
    ) -> Arc<Self> {
        let graph: node::Scene = serde_json::from_str(desc).unwrap_or_else(|e| {
            log::error!("error during scene loading:{:}", e);
            std::process::exit(-1);
        });
        let loader = SceneLoader::new(device, Rc::new(graph), file_resolver);
        loader.load()
    }
    pub fn env_map(&self, w: Expr<Float3>, evals: &Evaluators) -> Color {
        // TODO: fix this
        let (theta, phi) = xyz_to_spherical(w);
        let u = phi / (2.0 * PI);
        let v = theta / PI;
        let si = var!(SurfaceInteraction);
        si.set_geometry(zeroed::<SurfaceLocalGeometry>().set_uv(make_float2(u, v)));
        evals
            .texture
            .evaluate_color(self.env_map.var().read(0), si.load())
    }
    pub fn si_from_hitinfo(
        &self,
        inst_id: Expr<u32>,
        prim_id: Expr<u32>,
        bary: Expr<Float2>,
    ) -> SurfaceInteractionExpr {
        let shading_triangle = self.meshes.shading_triangle(inst_id, prim_id);
        let p = shading_triangle.p(bary);
        let n = shading_triangle.n(bary);
        let uv = shading_triangle.tc(bary);
        let geometry = SurfaceLocalGeometryExpr::new(
            p,
            shading_triangle.ng(),
            n,
            uv,
            Float3Expr::zero(),
            Float3Expr::zero(),
        );
        SurfaceInteractionExpr::new(
            inst_id,
            prim_id,
            bary,
            geometry,
            FrameExpr::from_n(n),
            Bool::from(true),
        )
    }
    pub fn intersect(&self, ray: Expr<Ray>) -> Expr<SurfaceInteraction> {
        let ro = ray.o();
        let rd = ray.d();
        let rtx_ray = rtx::RayExpr::new(ro, ray.t_min(), rd, ray.t_max());

        let hit = self.meshes.accel.var().query_all(
            rtx_ray,
            255,
            rtx::RayQuery {
                on_triangle_hit: |candidate: rtx::TriangleCandidate| {
                    if_!(
                        (candidate.inst().cmpne(ray.exclude0().x())
                            | candidate.prim().cmpne(ray.exclude0().y()))
                            & (candidate.inst().cmpne(ray.exclude1().x())
                                | candidate.prim().cmpne(ray.exclude1().y())),
                        {
                            candidate.commit();
                        }
                    );
                },
                on_procedural_hit: |_| {},
            },
        );
        // cpu_dbg!(hit);
        if_!(hit.triangle_hit(), {
            let inst_id = hit.inst_id();
            let prim_id = hit.prim_id();
            let bary = hit.bary();
            self.si_from_hitinfo(inst_id, prim_id, bary)
        }, else {
            zeroed::<SurfaceInteraction>().set_valid(Bool::from(false))
        })
    }
    pub fn occlude(&self, ray: Expr<Ray>) -> Expr<bool> {
        let ro = ray.o();
        let rd = ray.d();
        let rtx_ray = rtx::RayExpr::new(ro, ray.t_min(), rd, ray.t_max());

        let hit = self.meshes.accel.var().query_any(
            rtx_ray,
            255,
            rtx::RayQuery {
                on_triangle_hit: |candidate: rtx::TriangleCandidate| {
                    if_!(
                        (candidate.inst().cmpne(ray.exclude0().x())
                            | candidate.prim().cmpne(ray.exclude0().y()))
                            & (candidate.inst().cmpne(ray.exclude1().x())
                                | candidate.prim().cmpne(ray.exclude1().y())),
                        {
                            candidate.commit();
                        }
                    );
                },
                on_procedural_hit: |_| {},
            },
        );
        // cpu_dbg!(ray);
        // cpu_dbg!(hit);
        !hit.miss()
    }
}

struct SceneLoader {
    device: Device,
    graph: Rc<node::Scene>,
    named_surfaces: HashMap<String, (TagIndex, Option<TagIndex>)>,
    surfaces: PolymorphicBuilder<PolyKey, dyn Surface>,
    textures: PolymorphicBuilder<PolyKey, dyn Texture>,
    lights: PolymorphicBuilder<PolyKey, dyn Light>,
    light_ids_to_lights: Vec<TagIndex>,
    area_lights: Vec<TagIndex>,
    texture_cache: HashMap<String, u32>,
    bindless_texture: BindlessArray,
    mesh_cache: HashMap<String, (Arc<TriangleMesh>, u32)>,
    file_resolver: Arc<dyn FileResolver + Send + Sync>,
    meshes: Vec<Box<MeshBuffer>>,
    emission_power: HashMap<TagIndex, f32>,
    instances: Vec<MeshInstance>,
    light_weights: Vec<f32>,
    camera: Option<Box<dyn Camera>>,
}

impl SceneLoader {
    fn new(
        device: Device,
        graph: Rc<node::Scene>,
        file_resolver: Arc<dyn FileResolver + Send + Sync>,
    ) -> Self {
        Self {
            camera: None,
            device: device.clone(),
            graph,
            surfaces: PolymorphicBuilder::new(device.clone()),
            textures: PolymorphicBuilder::new(device.clone()),
            lights: PolymorphicBuilder::new(device.clone()),
            file_resolver,
            mesh_cache: HashMap::new(),
            texture_cache: HashMap::new(),
            bindless_texture: device.create_bindless_array(65536),
            named_surfaces: HashMap::new(),
            meshes: Vec::new(),
            emission_power: HashMap::new(),
            light_weights: Vec::new(),
            instances: Vec::new(),
            area_lights: Vec::new(),
            light_ids_to_lights: Vec::new(),
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
    fn load_transform(&self, t: &node::Transform, is_camera: bool) -> AffineTransform {
        match t {
            node::Transform::TRS(trs) => {
                let trs = *trs;
                let mut m = glam::Mat4::IDENTITY;
                let node::TRS {
                    translate: t,
                    rotate: r,
                    scale: s,
                    coord_sys,
                } = trs;
                let (t, r, s) = (t.into(), r.into(), s.into());
                let r: glam::Vec3 = r;
                let r = glam::vec3(r.x.to_radians(), r.y.to_radians(), r.z.to_radians());
                if !is_camera {
                    m = glam::Mat4::from_scale(s) * m;
                }
                if coord_sys == CoordinateSystem::Akari {
                    m = glam::Mat4::from_axis_angle(glam::vec3(0.0, 0.0, 1.0), r[2]) * m;
                    m = glam::Mat4::from_axis_angle(glam::vec3(1.0, 0.0, 0.0), r[0]) * m;
                    m = glam::Mat4::from_axis_angle(glam::vec3(0.0, 1.0, 0.0), r[1]) * m;
                    m = glam::Mat4::from_translation(t) * m;
                } else if coord_sys == CoordinateSystem::Blender {
                    if is_camera {
                        // blender camera starts pointing at (0,0,-1) a.k.a down
                        m = glam::Mat4::from_axis_angle(glam::vec3(1.0, 0.0, 0.0), -PI / 2.0) * m;
                    }
                    m = glam::Mat4::from_axis_angle(glam::vec3(1.0, 0.0, 0.0), r[0]) * m;
                    m = glam::Mat4::from_axis_angle(glam::vec3(0.0, 0.0, 1.0), -r[1]) * m;
                    m = glam::Mat4::from_axis_angle(glam::vec3(0.0, 1.0, 0.0), r[2]) * m;
                    m = glam::Mat4::from_translation(glam::vec3(t.x, t.z, -t.y)) * m;
                } else {
                    unreachable!()
                }

                AffineTransform::from_matrix(&m)
            }
            node::Transform::LookAt(node::LookAt { eye, center, up }) => {
                AffineTransform::from_matrix(
                    &glam::Mat4::look_at_rh(
                        glam::Vec3::from(*eye).into(),
                        glam::Vec3::from(*center).into(),
                        glam::Vec3::from(*up).into(),
                    )
                    .inverse(),
                )
            }
        }
    }
    fn load_texture(&mut self, node: &node::Texture) -> TagIndex {
        match node {
            node::Texture::Float { value: v } => {
                let tex = ConstFloatTexture { value: *v };
                self.textures.push(PolyKey::Simple("float".into()), tex)
            }
            node::Texture::SRgbLinear { values } => {
                let tex = ConstRgbTexture {
                    rgb: Float3::new(values[0], values[1], values[2]),
                };
                self.textures.push(PolyKey::Simple("srgb".into()), tex)
            }
            node::Texture::SRgb { values } => {
                let srgb: glam::Vec3 = (*values).into();
                let linear = glam_srgb_to_linear(srgb);
                let tex = ConstRgbTexture { rgb: linear.into() };
                self.textures.push(PolyKey::Simple("srgb".into()), tex)
            }
            node::Texture::SRgbU8 { values } => {
                let srgb: glam::Vec3 =
                    [values[0] as f32, values[1] as f32, values[2] as f32].into();
                let srgb = srgb / 255.0;
                let linear = glam_srgb_to_linear(srgb);
                let tex = ConstRgbTexture { rgb: linear.into() };
                self.textures.push(PolyKey::Simple("srgb".into()), tex)
            }
            node::Texture::Image { path, colorspace } => {
                assert_eq!(colorspace, "srgb");
                let mut file = self
                    .file_resolver
                    .resolve(&PathBuf::from(path))
                    .unwrap_or_else(|| panic!("cannot resolve path {}", path));
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes).unwrap();
                let img = image::io::Reader::new(Cursor::new(bytes))
                    .with_guessed_format()
                    .unwrap()
                    .decode()
                    .unwrap()
                    .to_rgba8();
                let pixels = img.pixels().map(|p| p.0).collect::<Vec<_>>();
                let tex = self.device.create_tex2d::<Float4>(
                    PixelStorage::Byte4,
                    img.width(),
                    img.height(),
                    1,
                );
                tex.view(0).copy_from(&pixels);
                let index = self.texture_cache.len() as u32;
                self.bindless_texture.set_tex2d(
                    index as usize,
                    &tex,
                    luisa::Sampler {
                        filter: luisa::SamplerFilter::Point,
                        address: luisa::SamplerAddress::Repeat,
                    },
                );
                self.texture_cache.insert(path.clone(), index);
                self.textures
                    .push(PolyKey::Simple("image".into()), ImageRgbTexture { index })
            }
        }
    }
    fn load_surface_from_name(&mut self, name: &String) -> (TagIndex, Option<TagIndex>) {
        if self.named_surfaces.contains_key(name) {
            self.named_surfaces.get(name).unwrap().clone()
        } else {
            let graph = self.graph.clone();
            let bsdf = graph.bsdfs.get(name).unwrap_or_else(|| {
                log::error!("bsdf {} is not defined", name);
                std::process::exit(-1);
            });
            let bsdf = self.load_surface(bsdf);
            self.named_surfaces.insert(name.clone(), bsdf.clone());
            bsdf
        }
    }
    fn load_surface(&mut self, node: &node::Bsdf) -> (TagIndex, Option<TagIndex>) {
        match node {
            node::Bsdf::Diffuse { color } => {
                let color = self.load_texture(color);
                let i = self.surfaces.push(
                    PolyKey::Simple("diffuse".into()),
                    DiffuseSurface { reflectance: color },
                );
                (i, None)
            }
            node::Bsdf::Glass {
                ior,
                kr,
                kt,
                roughness,
                ..
            } => {
                let kr = self.load_texture(kr);
                let kt = self.load_texture(kt);
                let roughness = self.load_texture(roughness);
                let ior = *ior;
                let i = self.surfaces.push(
                    PolyKey::Simple("diffuse".into()),
                    GlassSurface {
                        eta: ior,
                        kr,
                        kt,
                        roughness,
                    },
                );
                (i, None)
            }
            node::Bsdf::Principled {
                color,
                subsurface: _,
                subsurface_radius: _,
                subsurface_color: _,
                subsurface_ior: _,
                metallic,
                specular_tint: _,
                roughness,
                anisotropic: _,
                anisotropic_rotation: _,
                sheen: _,
                sheen_tint: _,
                clearcoat,
                clearcoat_roughness,
                ior,
                transmission,
                emission,
            } => {
                let color = self.load_texture(color);
                let metallic = self.load_texture(metallic);
                let roughness = self.load_texture(roughness);
                let transmission = self.load_texture(transmission);
                let clearcoat = self.load_texture(clearcoat);
                let clearcoat_roughness = self.load_texture(clearcoat_roughness);
                let bsdf_id = self.surfaces.push(
                    PolyKey::Simple("principled".into()),
                    PrincipledSurface {
                        color,
                        metallic,
                        roughness,
                        clearcoat,
                        clearcoat_roughness,
                        eta: *ior,
                        transmission,
                    },
                );
                let emission = self.load_texture(emission);
                (bsdf_id, Some(emission))
            }
            _ => todo!(),
        }
    }
    fn estimate_power(&mut self, tex_id: TagIndex) -> f32 {
        if let Some(power) = self.emission_power.get(&tex_id) {
            return *power;
        }
        let tex = self.textures.get(tex_id);
        let power = if let Some(tex) = tex.downcast_ref::<ConstRgbTexture>() {
            0.2126 * tex.rgb.x + 0.7152 * tex.rgb.y + 0.0722 * tex.rgb.z
        } else if let Some(tex) = tex.downcast_ref::<ConstFloatTexture>() {
            tex.value
        } else {
            unreachable!()
        };
        self.emission_power.insert(tex_id, power);
        power
    }
    fn load_shape(&mut self, node: &node::Shape) -> MeshInstance {
        match node {
            node::Shape::Mesh {
                path,
                bsdf,
                transform,
            } => {
                let (model, id) = if let Some((cache, id)) = self.mesh_cache.get(path) {
                    (cache.clone(), *id)
                } else {
                    let mut file = self.resolve_file(path);
                    let model = Arc::new(TriangleMesh::decode(&mut file).unwrap());
                    let mesh_id = self.meshes.len() as u32;
                    let mesh = Box::new(MeshBuffer::new(self.device.clone(), &model));
                    self.mesh_cache
                        .insert(path.clone(), (model.clone(), mesh_id));
                    self.meshes.push(mesh);
                    (model, mesh_id)
                };
                let transform = if let Some(transform) = transform {
                    self.load_transform(transform, false)
                } else {
                    AffineTransform::from_matrix(&glam::Mat4::IDENTITY)
                };
                let determinant = glam::Mat3::from(transform.m3).determinant().abs();
                let (surface, emission) = self.load_surface_from_name(bsdf);
                let mut light = None;
                if let Some(emission) = emission {
                    let power = self.estimate_power(emission);
                    if power > 0.0 {
                        let mesh = &mut self.meshes[id as usize];
                        let areas = model.areas();
                        if mesh.area_sampler.is_none() {
                            mesh.build_area_sampler(self.device.clone(), &areas);
                        }
                        let weight = areas.par_iter().sum::<f32>() * determinant;
                        let light_id = self.light_weights.len() as u32;

                        let i = self.lights.push(
                            PolyKey::Simple("area".into()),
                            AreaLight {
                                light_id,
                                instance_id: self.instances.len() as u32,
                                area_sampling_index: u32::MAX,
                                emission,
                            },
                        );
                        light = Some(i);
                        self.light_ids_to_lights.push(i);
                        self.area_lights.push(i);
                        self.light_weights.push(weight * power);
                    }
                }
                let mesh = &self.meshes[id as usize];
                let instance = MeshInstance {
                    geom_id: id,
                    transform,
                    has_normals: mesh.has_normals,
                    has_texcoords: mesh.has_texcoords,
                    light: light.unwrap_or(TagIndex::INVALID),
                    surface,
                };
                self.instances.push(instance);
                instance
            }
            _ => unimplemented!(),
        }
    }
    fn load(mut self) -> Arc<Scene> {
        self.camera = Some(match self.graph.camera {
            node::Camera::Perspective {
                res,
                fov,
                lens_radius: _,
                focal: _,
                transform,
            } => Box::new(PerspectiveCamera::new(
                self.device.clone(),
                Uint2::new(res.0, res.1),
                self.load_transform(&transform, true),
                fov.to_radians() as f32,
                0.0,
                1.0,
            )),
        });
        let graph = self.graph.clone();
        for node in graph.shapes.iter() {
            let _ = self.load_shape(node);
        }
        let env_map = if let Some(tex) = self.graph.environment.clone() {
            self.load_texture(&tex)
        } else {
            self.load_texture(&node::Texture::Float { value: 0.0 })
        };
        let SceneLoader {
            meshes,
            textures,
            surfaces,
            mut instances,
            mut lights,
            light_ids_to_lights,
            light_weights,
            area_lights,
            device,
            camera,
            bindless_texture: image_textures,
            ..
        } = self;
        let env_map = device.create_buffer_from_slice(&[env_map]);
        let textures = textures.build();
        let surfaces = surfaces.build();
        let meshes = meshes.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
        let meshes = Arc::new(MeshAggregate::new(device.clone(), &meshes, &mut instances));
        log::info!("Area light count: {}", area_lights.len());
        for i in area_lights {
            let mut area = lights.get_mut(i).downcast_mut::<AreaLight>().unwrap();
            area.area_sampling_index = meshes.mesh_id_to_area_samplers[&area.instance_id];
        }
        let lights = lights.build();
        let light_distribution = Box::new(WeightedLightDistribution::new(
            device.clone(),
            &light_weights,
        ));
        let light_ids_to_lights = device.create_buffer_from_slice(&light_ids_to_lights);
        let lights = LightAggregate {
            light_distribution,
            lights,
            light_ids_to_lights,
            meshes: meshes.clone(),
        };
        Arc::new(Scene {
            textures,
            surfaces,
            lights,
            meshes,
            camera: camera.unwrap(),
            device,
            image_textures,
            env_map,
        })
    }
}
