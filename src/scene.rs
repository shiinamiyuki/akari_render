use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File;
use std::sync::Arc;
use std::{path::PathBuf, rc::Rc};

use crate::camera::PerspectiveCamera;
use crate::color::glam_srgb_to_linear;
use crate::light::area::AreaLight;
use crate::light::WeightedLightDistribution;
use crate::scenegraph::node::CoordinateSystem;
use crate::surface::diffuse::{DiffuseBsdf, DiffuseSurface};
use crate::texture::{ConstFloatTexture, ConstRgbTexture};
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
    pub lights: Polymorphic<PolyKey, dyn Light>,
    pub light_distribution: Box<dyn LightDistribution>,
    pub meshes: MeshAggregate,
}

impl Scene {
    pub fn intersect(&self, ray: Expr<Ray>) -> Expr<SurfaceInteraction> {
        let ro = ray.o();
        let rd = ray.d();
        let rtx_ray = rtx::RayExpr::new(ro, ray.t_min(), rd, ray.t_max());
        let hit = self.meshes.accel.var().trace_closest(rtx_ray);
        if_!(hit.valid(), {
            let inst_id = hit.inst_id();
            let prim_id = hit.prim_id();
            let bary = make_float2(hit.u(), hit.v());
            let shading_triangle = self.meshes.shading_triangle(inst_id, prim_id);
            let p = shading_triangle.p(bary);
            let n = shading_triangle.n(bary);
            let uv = shading_triangle.tc(bary);
            let geometry = SurfaceLocalGeometryExpr::new(p, shading_triangle.ng(), n, uv, Float3Expr::zero(), Float3Expr::zero());
            SurfaceInteractionExpr::new(geometry, bary, prim_id, inst_id, FrameExpr::from_n(n), shading_triangle, Bool::from(true))
        }, else {
            zeroed::<SurfaceInteraction>().set_valid(Bool::from(false))
        })
    }
    pub fn occlude(&self, ray: Expr<Ray>) -> Expr<bool> {
        let ro = ray.o();
        let rd = ray.d();
        let rtx_ray = rtx::RayExpr::new(ro, ray.t_min(), rd, ray.t_max());
        self.meshes.accel.var().trace_any(rtx_ray)
    }
}

struct SceneLoader {
    device: Device,
    parent_path: PathBuf,
    graph: Rc<node::Scene>,
    named_surfaces: HashMap<String, (TagIndex, Option<TagIndex>)>,
    surfaces: PolymorphicBuilder<PolyKey, dyn Surface>,
    textures: PolymorphicBuilder<PolyKey, dyn Texture>,
    lights: PolymorphicBuilder<PolyKey, dyn Light>,
    area_lights: Vec<TagIndex>,
    texture_cache: HashMap<String, TagIndex>,
    mesh_cache: HashMap<String, (Arc<TriangleMesh>, u32)>,
    file_resolver: Arc<dyn FileResolver + Send + Sync>,
    meshes: Vec<Box<MeshBuffer>>,
    emission_power: HashMap<TagIndex, f32>,
    instances: Vec<MeshInstance>,
    light_weights: Vec<f32>,
    camera: Option<Buffer<PerspectiveCamera>>,
}

impl SceneLoader {
    fn new(
        device: Device,
        parent_path: PathBuf,
        graph: Rc<node::Scene>,
        file_resolver: Arc<dyn FileResolver + Send + Sync>,
    ) -> Self {
        Self {
            camera: None,
            device: device.clone(),
            parent_path,
            graph,
            surfaces: PolymorphicBuilder::new(device.clone()),
            textures: PolymorphicBuilder::new(device.clone()),
            lights: PolymorphicBuilder::new(device.clone()),
            file_resolver,
            mesh_cache: HashMap::new(),
            texture_cache: HashMap::new(),
            named_surfaces: HashMap::new(),
            meshes: Vec::new(),
            emission_power: HashMap::new(),
            light_weights: Vec::new(),
            instances: Vec::new(),
            area_lights: Vec::new(),
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
    fn load_color(&mut self, node: &node::Texture) -> TagIndex {
        match node {
            node::Texture::Float(v) => {
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
            node::Texture::Image { .. } => {
                todo!()
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
                let color = self.load_color(color);
                let i = self.surfaces.push(
                    PolyKey::Simple("diffuse".into()),
                    DiffuseSurface { reflectance: color },
                );
                (i, None)
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
    fn load_shape(&mut self, node: &node::Shape) -> luisa::Result<MeshInstance> {
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
                    let model = Arc::new({ TriangleMesh::decode(&mut file).unwrap() });
                    let mesh_id = self.meshes.len() as u32;
                    let mesh = Box::new(MeshBuffer::new(self.device.clone(), &model)?);
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
                if let Some(emission) = emission {
                    let power = self.estimate_power(emission);
                    let mesh = &mut self.meshes[id as usize];
                    let areas = model.areas();
                    if mesh.area_sampler.is_none() {
                        mesh.build_area_sampler(self.device.clone(), &areas)?;
                    }
                    let weight = areas.par_iter().sum::<f32>() * determinant;
                    self.light_weights.push(weight * power);
                    let i = self.lights.push(
                        PolyKey::Simple("area".into()),
                        AreaLight {
                            instance_id: self.instances.len() as u32,
                            area_sampling_index: u32::MAX,
                        },
                    );
                    self.area_lights.push(i);
                } else {
                    self.light_weights.push(0.0);
                }
                let instance = MeshInstance {
                    geom_id: id,
                    transform,
                    normal_index: u32::MAX, // set these later
                    texcoord_index: u32::MAX,
                    emission_tex: emission.unwrap_or(TagIndex::INVALID),
                    surface,
                };
                self.instances.push(instance);
                Ok(instance)
            }
            _ => unimplemented!(),
        }
    }
    fn load(mut self) -> luisa::Result<Scene> {
        self.camera = Some(match self.graph.camera {
            node::Camera::Perspective {
                res,
                fov,
                lens_radius: _,
                focal: _,
                transform,
            } => PerspectiveCamera::new(
                self.device.clone(),
                Uint2::new(res.0, res.1),
                self.load_transform(&transform, true),
                fov.to_radians() as f32,
                0.0,
                1.0,
            )?,
        });
        let graph = self.graph.clone();
        for node in graph.shapes.iter() {
            let _ = self.load_shape(node)?;
        }
        let SceneLoader {
            meshes,
            textures,
            surfaces,
            mut instances,
            mut lights,
            light_weights,
            area_lights,
            device,
            ..
        } = self;
        let textures = textures.build()?;
        let surfaces = surfaces.build()?;
        let meshes = meshes.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
        let meshes = MeshAggregate::new(device.clone(), &meshes, &mut instances)?;
        for i in area_lights {
            let mut area = lights.get_mut(i).downcast_mut::<AreaLight>().unwrap();
            area.area_sampling_index = meshes.mesh_id_to_area_samplers[&area.instance_id];
        }
        let lights = lights.build()?;
        let light_distribution = Box::new(WeightedLightDistribution::new(device.clone(), &light_weights)?);
        Ok(Scene {
            textures,
            surfaces,
            lights,
            light_distribution,
            meshes,
        })
    }
}
