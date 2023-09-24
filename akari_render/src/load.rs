use crate::{
    camera::{Camera, PerspectiveCamera},
    geometry::AffineTransform,
    light::{area::AreaLight, Light, LightAggregate, WeightedLightDistribution},
    mesh::{MeshAggregate, MeshBuffer, MeshInstance, TriangleMesh},
    nodes::CoordinateSystem,
    scene::Scene,
    svm::{
        compiler::{CompilerDriver, SvmCompileContext},
        ShaderRef, Svm,
    },
    util::{binserde::Decode, FileResolver, LocalFileResolver},
    *,
};
use akari_nodegraph::{parse, Node, NodeGraph, NodeGraphDesc, NodeId, NodeProxy};
use image::io::Reader as ImageReader;

use lazy_static::lazy_static;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};
pub struct SceneLoader {
    device: Device,
    images: HashMap<String, image::DynamicImage>,
    textures: Vec<Tex2d<Float4>>,
    texture_heap: BindlessArray,
    node_to_mesh: HashMap<NodeId, (usize, Arc<TriangleMesh>)>,
    meshes: Vec<Arc<TriangleMesh>>,
    mesh_areas: Vec<Vec<f32>>,
    mesh_buffers: Vec<MeshBuffer>,
    graph: NodeGraph,
    camera: Option<Arc<dyn Camera>>,
    root: NodeId,
    surface_shader_compiler: CompilerDriver,
    material_nodes: Vec<NodeId>,
    light_nodes: Vec<NodeId>,
    lights: PolymorphicBuilder<(), dyn Light>,
    nodes_to_surface_shader: HashMap<NodeId, ShaderRef>,
    path_sampler_to_idx: HashMap<(String, TextureSampler), usize>,
}
fn load_buffer<T>(file_resolver: &dyn FileResolver, path: impl AsRef<Path>) -> Vec<T>
where
    Vec<T>: Decode,
{
    log::info!("Loading buffer: {}", path.as_ref().display());
    let mut file = file_resolver.resolve(path.as_ref()).unwrap_or_else(|| {
        panic!(
            "Failed to resolve file: {}",
            path.as_ref().display().to_string()
        )
    });
    Vec::<T>::decode(&mut file).unwrap()
}
impl SceneLoader {
    pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Scene> {
        log::info!("Loading scene: {}", path.as_ref().display());
        let graph = load_scene_graph(&path);
        log::info!("Loaded scene graph");
        let loader = Self::preload(device, graph);
        let scene = Arc::new(loader.do_load());
        log::info!("Loaded scene: {}", path.as_ref().display());
        scene
    }
    fn estimate_emission_tex_intensity_fast(&self, emission: &NodeId) -> Option<f32> {
        let emission = &self.graph.nodes[emission];
        let ty = emission.ty()?;
        if ty != nodes::SpectralUplift::ty() {
            return None;
        }
        let up = emission
            .proxy::<nodes::SpectralUplift>(&self.graph)
            .unwrap();
        let rgb = &self.graph.nodes[&up.in_rgb.as_node()?.from];
        let ty = rgb.ty()?;
        if ty == nodes::RGB::ty() {
            let rgb = rgb.proxy::<nodes::RGB>(&self.graph).unwrap();
            let r = *rgb.in_r.as_value()?;
            let g = *rgb.in_g.as_value()?;
            let b = *rgb.in_b.as_value()?;
            Some(r.max(g).max(b) as f32)
        } else {
            None
        }
    }
    fn compute_mesh_area(&mut self, geom_id: usize) {
        if !self.mesh_areas[geom_id].is_empty() {
            return;
        }
        self.mesh_areas[geom_id] = self.meshes[geom_id].areas();
    }
    fn mesh_total_area(&mut self, geom_id: usize) -> f32 {
        self.compute_mesh_area(geom_id);
        self.mesh_areas[geom_id].par_iter().sum()
    }
    fn estimate_surface_emission_power(
        &mut self,
        instance: &MeshInstance,
        surface: &NodeId,
    ) -> f32 {
        let surface = &self.graph.nodes[surface];
        let ty = surface.ty().unwrap();
        let (emission, strength) = if ty == nodes::PrincipledBsdf::ty() {
            let bsdf = surface.proxy::<nodes::PrincipledBsdf>(&self.graph).unwrap();
            (
                Some(bsdf.in_emission.as_node().unwrap().from.clone()),
                Some(1.0f32),
            )
        } else if ty == nodes::Emission::ty() {
            let emission = Some(
                surface
                    .proxy::<nodes::Emission>(&self.graph)
                    .unwrap()
                    .in_color
                    .as_node()
                    .unwrap()
                    .from
                    .clone(),
            );
            let strength = surface
                .proxy::<nodes::Emission>(&self.graph)
                .unwrap()
                .in_strength
                .as_value()
                .map(|n| *n as f32);
            (emission, strength)
        } else {
            (None, None)
        };
        if let Some(emission) = &emission {
            if let Some(strength) = strength {
                if let Some(power) = self.estimate_emission_tex_intensity_fast(emission) {
                    if power == 0.0 {
                        return 0.0;
                    }

                    let geom_id = instance.geom_id as usize;
                    let area = self.mesh_total_area(geom_id);
                    let transform: glam::Mat3 = instance.transform.m3.into();
                    let det = transform.determinant();
                    // dbg!(power, area, det, strength);
                    return power * area * det.abs() * strength;
                }
            }
            todo!()
        } else {
            0.0
        }
    }
    fn load_transform(&self, node: &NodeId, is_camera: bool) -> AffineTransform {
        let node = &self.graph.nodes[node];
        let ty = node.ty().unwrap();
        if ty == nodes::TRS::ty() {
            let trs = node.proxy::<nodes::TRS>(&self.graph).unwrap();
            let coord_sys = trs.in_coordinate_system;
            let translate = trs
                .in_translation
                .iter()
                .map(|n| *n.as_value().unwrap() as f32)
                .collect::<Vec<_>>();
            let rotate = trs
                .in_rotation
                .iter()
                .map(|n| *n.as_value().unwrap() as f32)
                .collect::<Vec<_>>();
            let scale = trs
                .in_scale
                .iter()
                .map(|n| *n.as_value().unwrap() as f32)
                .collect::<Vec<_>>();
            let mut m = glam::Mat4::IDENTITY;
            let t = glam::vec3(translate[0], translate[1], translate[2]);
            let r = glam::vec3(rotate[0], rotate[1], rotate[2]);
            let s = glam::vec3(scale[0], scale[1], scale[2]);
            // dbg!(t,r,s);
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
        } else if ty == nodes::MatrixTransform::ty() {
            let mat = node.proxy::<nodes::MatrixTransform>(&self.graph).unwrap();
            let mat = mat
                .in_matrix
                .iter()
                .map(|n| *n.as_value().unwrap() as f32)
                .collect::<Vec<_>>();
            let m = glam::Mat4::from_cols_slice(&mat).transpose();
            AffineTransform::from_matrix(&m)
        } else {
            panic!("Unsupported transform type: {}", ty);
        }
    }
    fn load_camera(&self, node_id: &NodeId) -> Arc<dyn Camera> {
        let node = &self.graph.nodes[node_id];
        let ty = node.ty().unwrap();
        if ty == nodes::PerspectiveCamera::ty() {
            let cam = node.proxy::<nodes::PerspectiveCamera>(&self.graph).unwrap();
            let transform = self.load_transform(&cam.in_transform.as_node().unwrap().from, true);
            let fov = (*cam.in_fov.as_value().unwrap() as f32).to_radians();
            let focal_distance = *cam.in_focal_distance.as_value().unwrap() as f32;
            let fstop = *cam.in_fstop.as_value().unwrap() as f32;
            let lens_radius = focal_distance / (2.0 * fstop);
            let width = *cam.in_width.as_value().unwrap() as u32;
            let height = *cam.in_height.as_value().unwrap() as u32;
            let cam = Arc::new(PerspectiveCamera::new(
                self.device.clone(),
                Uint2::new(width, height),
                transform,
                fov,
                lens_radius,
                focal_distance,
            ));
            cam
        } else {
            todo!()
        }
    }
    fn load_instance(&self, node_id: &NodeId) -> (NodeId, MeshInstance) {
        let node = &self.graph.nodes[node_id];
        let instance = node.proxy::<nodes::Instance>(&self.graph).unwrap();
        let mat = instance.in_material.as_node().unwrap();
        let mat = self.graph.nodes[&mat.from]
            .proxy::<nodes::MaterialOutput>(&self.graph)
            .unwrap();
        let surface_id = &mat.in_surface.as_node().unwrap().from;
        let surface = self.nodes_to_surface_shader[surface_id];
        let transform = self.load_transform(&instance.in_transform.as_node().unwrap().from, false);
        let geometry_node_id = &instance.in_geometry.as_node().unwrap().from;
        let geometry_node = &self.graph.nodes[geometry_node_id];
        if geometry_node.ty().unwrap() == nodes::Mesh::ty() {
            let geom_id = self.node_to_mesh[geometry_node_id].0;
            let mesh_buffer = &self.mesh_buffers[geom_id];
            (
                surface_id.clone(),
                MeshInstance {
                    geom_id: geom_id as u32,
                    transform,
                    light: TagIndex::INVALID,
                    surface,
                    has_normals: mesh_buffer.has_normals,
                    has_uvs: mesh_buffer.has_uvs,
                    has_tangents: mesh_buffer.has_tangents,
                },
            )
        } else {
            todo!()
        }
    }
    fn do_load(mut self) -> Scene {
        self.mesh_areas.resize(self.mesh_buffers.len(), vec![]);
        for mat_node in &self.material_nodes {
            let node = &self.graph.nodes[mat_node];
            let mat = node.proxy::<nodes::MaterialOutput>(&self.graph).unwrap();
            if let Some(bsdf) = mat.in_surface.as_node() {
                let shader = self.surface_shader_compiler.compile(
                    &bsdf.from,
                    SvmCompileContext {
                        images: &self.path_sampler_to_idx,
                        graph: &self.graph,
                    },
                );
                self.nodes_to_surface_shader
                    .insert(bsdf.from.clone(), shader);
            } else {
                panic!("No bsdf node found");
            }
        }
        for light_node in &self.light_nodes {
            let node = &self.graph.nodes[light_node];
            let light = node.proxy::<nodes::LightOutput>(&self.graph).unwrap();
            let surface = &light.in_surface.as_node().unwrap().from;
            let shader = self.surface_shader_compiler.compile(
                surface,
                SvmCompileContext {
                    images: &self.path_sampler_to_idx,
                    graph: &self.graph,
                },
            );
            self.nodes_to_surface_shader.insert(surface.clone(), shader);
        }
        let mut instances = vec![];
        let mut instance_surfaces = vec![];
        let scene_node = &self.graph.nodes[&self.root];
        let scene = scene_node.proxy::<nodes::Scene>(&self.graph).unwrap();
        {
            let camera = &scene.in_camera.as_node().unwrap().from;
            let camera = self.load_camera(camera);
            self.camera = Some(camera);
        }
        let instance_nodes = scene
            .in_instances
            .iter()
            .map(|n| n.as_node().unwrap().from.clone())
            .collect::<Vec<_>>();
        for instance in &instance_nodes {
            let (surface, instance) = self.load_instance(instance);
            instances.push(instance);
            instance_surfaces.push(surface);
        }
        let mut lights = vec![];
        let mut light_ids_to_lights = vec![];
        // now compute light emission power
        for (i, inst) in instances.iter_mut().enumerate() {
            let power = self.estimate_surface_emission_power(inst, &instance_surfaces[i]);

            if 0.0 < power && power <= 1e-4 {
                log::warn!(
                    "Light power too low: {}, power: {}",
                    instance_nodes[i].0,
                    power
                );
            }
            if power > 1e-4 {
                let light_id = lights.len();
                lights.push((i, power));

                let geom_id = inst.geom_id as usize;
                self.compute_mesh_area(geom_id);
                let mesh = &mut self.mesh_buffers[geom_id];
                if mesh.area_sampler.is_none() {
                    mesh.build_area_sampler(self.device.clone(), &self.mesh_areas[geom_id]);
                }
                let surface_shader = &self.nodes_to_surface_shader[&instance_surfaces[i]];
                let light_ref = self.lights.push(
                    (),
                    AreaLight {
                        light_id: light_id as u32,
                        instance_id: i as u32,
                        area_sampling_index: u32::MAX,
                        surface: *surface_shader,
                    },
                );
                light_ids_to_lights.push(light_ref);
                inst.light = light_ref;
            }
        }
        let area_light_count = lights.len();
        log::info!("{} mesh lights found", area_light_count);
        // now add other lights
        {
            // TODO: add other lights
        }
        log::info!(
            "Building accel for {} meshes, {} instances",
            self.meshes.len(),
            instances.len()
        );
        let mesh_aggregate = Arc::new(MeshAggregate::new(
            self.device.clone(),
            &self.mesh_buffers.iter().collect::<Vec<_>>(),
            &instances,
        ));
        for i in 0..area_light_count {
            let area = self
                .lights
                .get_mut(light_ids_to_lights[i])
                .downcast_mut::<AreaLight>()
                .unwrap();
            let instance = &instances[area.instance_id as usize];
            area.area_sampling_index = mesh_aggregate.mesh_id_to_area_samplers[&instance.geom_id];
        }
        let light_weights = lights.iter().map(|(_, power)| *power).collect::<Vec<_>>();
        let Self {
            device,
            lights,
            surface_shader_compiler,
            texture_heap,
            camera,
            ..
        } = self;
        log::info!(
            "Shader variant count: {}",
            surface_shader_compiler.variant_count()
        );
        let svm = Arc::new(Svm {
            device: device.clone(),
            surface_shaders: surface_shader_compiler.upload(&device),
            image_textures: texture_heap,
        });
        let lights = lights.build();
        let light_distribution = Box::new(WeightedLightDistribution::new(
            device.clone(),
            &light_weights,
        ));
        let light_aggregate = LightAggregate {
            light_distribution,
            lights,
            light_ids_to_lights: device.create_buffer_from_slice(&light_ids_to_lights),
            meshes: mesh_aggregate.clone(),
        };
        Scene {
            svm,
            lights: light_aggregate,
            meshes: mesh_aggregate.clone(),
            camera: camera.unwrap(),
            device,
            // env_map: todo!(),
        }
    }
    fn preload(device: Device, graph: NodeGraph) -> Self {
        let mut node_to_mesh = HashMap::new();
        let mut images_to_load: HashSet<String> = HashSet::new();
        let file_resolver = LocalFileResolver::new(vec![]);
        let mut root = None;
        let mut material_nodes = vec![];
        let mut light_nodes = vec![];
        let mut meshes = vec![];
        let mut mesh_buffers = vec![];
        let mut path_samplers: HashSet<(String, TextureSampler)> = HashSet::new();
        let sorted_node_ids = {
            let mut sorted_node_ids = graph.nodes.keys().collect::<Vec<_>>();
            sorted_node_ids.sort();
            sorted_node_ids
        };
        // let mut instance_nodes = vec![];
        for id in sorted_node_ids {
            let node = &graph.nodes[id];
            let ty = node.ty().unwrap();
            if ty == nodes::RGBImageTexture::ty() {
                let tex = node.proxy::<nodes::RGBImageTexture>(&graph).unwrap();
                let path = tex.in_path.as_value().unwrap();
                let sampler = sampler_from_rgb_image_tex_node(&tex);
                path_samplers.insert((path.clone(), sampler));
                images_to_load.insert(path.clone());
            } else if ty == nodes::Mesh::ty() {
                let mesh = node.proxy::<nodes::Mesh>(&graph).unwrap();
                let buffers = mesh
                    .in_buffers
                    .iter()
                    .map(|b| {
                        let from = &graph.nodes[&b.as_node().unwrap().from];
                        let buf = from.proxy::<nodes::Buffer>(&graph).unwrap();
                        let path = buf.in_path.as_value().unwrap().clone();
                        let name = buf.in_name.as_value().unwrap().clone();
                        (name, path)
                    })
                    .collect::<HashMap<_, _>>();
                log::info!("Loading mesh: {}", mesh.in_name.as_value().unwrap());
                let vertices = load_buffer::<[f32;3]>(&file_resolver, &buffers["vertices"]);
                let normals = load_buffer::<[f32;3]>(&file_resolver, &buffers["normals"]);
                let indices = load_buffer::<[u32;3]>(&file_resolver, &buffers["indices"]);
                let mut uvs = vec![];
                if let Some(uv) = buffers.get("uvs") {
                    uvs = load_buffer::<[f32;2]>(&file_resolver, uv);
                }
                let mut tangents = vec![];
                if let Some(tangent) = buffers.get("tangents") {
                    tangents = load_buffer::<[f32;3]>(&file_resolver, tangent);
                }
                let mut bitangent_signs = vec![];
                if let Some(bitangent_sign) = buffers.get("bitangent_signs") {
                    bitangent_signs = load_buffer::<u32>(&file_resolver, bitangent_sign);
                }
                let mesh = Arc::new(TriangleMesh {
                    name: mesh.in_name.as_value().unwrap().clone(),
                    vertices,
                    normals,
                    indices,
                    uvs,
                    tangents,
                    bitangent_signs,
                });
                let mesh_buffer = MeshBuffer::new(device.clone(), &mesh);
                let geom_id = mesh_buffers.len();
                mesh_buffers.push(mesh_buffer);
                meshes.push(mesh.clone());
                node_to_mesh.insert(id.clone(), (geom_id, mesh));
            } else if ty == nodes::Scene::ty() {
                assert!(root.is_none(), "Multiple scene nodes found");
                root = Some(id.clone());
            } else if ty == nodes::MaterialOutput::ty() {
                material_nodes.push(id.clone());
            } else if ty == nodes::LightOutput::ty() {
                light_nodes.push(id.clone());
            }
        }
        let root = root.unwrap_or_else(|| panic!("No scene node found"));
        log::info!(
            "Scene has {} meshes, {} image textures",
            meshes.len(),
            images_to_load.len()
        );
        let images = images_to_load
            .into_par_iter()
            .map(|path_s| {
                let path = PathBuf::from(&path_s);
                let path = path.canonicalize().unwrap();
                log::info!("Loading image: {}", path.display());
                let file = file_resolver.resolve(&path).unwrap_or_else(|| {
                    panic!("Failed to resolve file: {}", path.display().to_string())
                });
                let img = ImageReader::new(BufReader::new(file))
                    .with_guessed_format()
                    .unwrap()
                    .decode()
                    .unwrap();
                (path_s, img)
            })
            .collect::<HashMap<_, _>>();
        log::info!("Loaded {} images", images.len());
        let orded_image_paths = images.keys().collect::<Vec<_>>();
        let textures = {
            orded_image_paths
                .par_iter()
                .map(|path| {
                    let img = &images[*path];
                    let img = img.to_rgba8();
                    let tex =
                        device.create_tex2d(PixelStorage::Byte4, img.width(), img.height(), 1);
                    tex
                })
                .collect::<Vec<_>>()
        };
        let path_to_texture_idx = orded_image_paths
            .into_iter()
            .enumerate()
            .map(|(i, path)| (path.clone(), i))
            .collect::<HashMap<_, _>>();
        let (texture_heap, path_sampler_to_idx) = {
            let mut path_sampler_to_idx = HashMap::new();
            let mut to_be_commited_to_heap = vec![];
            for (path, sampler) in &path_samplers {
                let key = (path.clone(), sampler.clone());
                if let None = path_sampler_to_idx.get(&key) {
                    let idx = to_be_commited_to_heap.len();
                    to_be_commited_to_heap.push((path.clone(), sampler.clone()));
                    path_sampler_to_idx.insert(key, idx);
                }
            }
            let texture_heap = device.create_bindless_array(to_be_commited_to_heap.len().max(1));
            for (i, (path, sampler)) in to_be_commited_to_heap.into_iter().enumerate() {
                let idx = path_to_texture_idx[&path];
                texture_heap.emplace_tex2d_async(i, &textures[idx], sampler);
            }
            if !path_sampler_to_idx.is_empty() {
                texture_heap.update();
            }
            (texture_heap, path_sampler_to_idx)
        };
        log::info!(
            "Texture Heap contains total {} (texture, sampler) tuples",
            path_sampler_to_idx.len()
        );
        Self {
            device: device.clone(),
            images,
            node_to_mesh,
            meshes,
            mesh_buffers,
            graph,
            camera: None,
            root,
            texture_heap,
            textures,
            surface_shader_compiler: CompilerDriver::new(),
            material_nodes,
            nodes_to_surface_shader: HashMap::new(),
            light_nodes,
            mesh_areas: vec![],
            lights: PolymorphicBuilder::new(device.clone()),
            path_sampler_to_idx,
        }
    }
}

lazy_static! {
    static ref SCENE_DESC: NodeGraphDesc =
        serde_json::from_str(include_str!("nodes.json")).unwrap();
}
pub fn load_scene_graph<P: AsRef<Path>>(path: P) -> NodeGraph {
    let path = path.as_ref();
    let ext = path.extension().unwrap().to_str().unwrap();
    if ext == "akr" {
        let mut file = File::open(path).unwrap();
        let mut buf = String::new();
        file.read_to_string(&mut buf).unwrap();
        parse::parse(&buf, &SCENE_DESC).unwrap()
    } else if ext == "json" {
        let file = File::open(path).unwrap();
        serde_json::from_reader(file).unwrap()
    } else {
        panic!("Unsupported scene file format: {}", ext);
    }
}

pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Scene> {
    SceneLoader::load_from_path(device, path)
}

pub(crate) fn sampler_from_rgb_image_tex_node(tex: &nodes::RGBImageTexture) -> TextureSampler {
    let extension = tex.in_extension;
    let interp = tex.in_interpolation;
    let sampler = TextureSampler {
        address: match extension {
            nodes::TextureExtension::Repeat => SamplerAddress::Repeat,
            nodes::TextureExtension::Clip => SamplerAddress::Zero,
            nodes::TextureExtension::Mirror => SamplerAddress::Mirror,
            nodes::TextureExtension::Extend => SamplerAddress::Edge,
        },
        filter: match interp {
            nodes::TextureInterpolation::Linear => SamplerFilter::LinearLinear,
            nodes::TextureInterpolation::Closest => SamplerFilter::LinearPoint,
        },
    };
    sampler
}
