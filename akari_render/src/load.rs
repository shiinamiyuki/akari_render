use crate::{
    camera::{Camera, PerspectiveCamera},
    geometry::AffineTransform,
    light::{area::AreaLight, Light, LightAggregate, WeightedLightDistribution},
    mesh::{MeshAggregate, MeshBuffer, MeshInstance, TriangleMesh},
    node::{shader::Node, CoordinateSystem, Material, Ref, ShaderGraph},
    scene::Scene,
    svm::{
        compiler::{CompilerDriver, SvmCompileContext},
        ShaderRef, Svm,
    },
    util::{binserde::Decode, FileResolver, LocalFileResolver},
    *,
};
use image::io::Reader as ImageReader;

use std::{
    cell::RefCell,
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
    node_to_mesh: HashMap<Ref<node::Geometry>, (usize, Arc<TriangleMesh>)>,
    meshes: Vec<Arc<TriangleMesh>>,
    mesh_areas: RefCell<Vec<Vec<f32>>>,
    mesh_buffers: Vec<MeshBuffer>,
    graph: node::Scene,
    camera: Option<Arc<dyn Camera>>,
    surface_shader_compiler: CompilerDriver,
    lights: PolymorphicBuilder<(), dyn Light>,
    nodes_to_surface_shader: HashMap<Ref<Material>, ShaderRef>,
    path_sampler_to_idx: HashMap<(String, TextureSampler), usize>,
}

fn load_buffer<T>(file_resolver: &dyn FileResolver, buffer: &node::Buffer) -> Vec<T>
where
    Vec<T>: Decode,
{
    match buffer {
        node::Buffer::External(path) => {
            log::info!("Loading buffer: {}", path);
            let mut file = file_resolver
                .resolve(&Path::new(path))
                .unwrap_or_else(|| panic!("Failed to resolve file: {}", path));
            Vec::<T>::decode(&mut file).unwrap()
        }
        node::Buffer::Internal(data) => {
            let mut data = data.as_slice();
            Vec::<T>::decode(&mut data).unwrap()
        }
    }
}
#[deny(dead_code)]
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
    fn estimate_emission_tex_intensity_fast(
        &self,
        emission: &Ref<Node>,
        shader_graph: &ShaderGraph,
    ) -> Option<f32> {
        let emission = &shader_graph[emission];
        match emission {
            Node::SpectralUplift(rgb) => {
                self.estimate_emission_tex_intensity_fast(rgb, shader_graph)
            }
            Node::Float(x) => return Some(*x),
            Node::Float3(x) => return Some(x.iter().copied().reduce(|a, b| a.max(b)).unwrap()),
            Node::Rgb { value: rgb, .. } => {
                Some(rgb.iter().copied().reduce(|a, b| a.max(b)).unwrap())
            }
            _ => return None,
        }
    }
    fn compute_mesh_area(&self, geom_id: usize) {
        let mut mesh_areas = self.mesh_areas.borrow_mut();
        if !mesh_areas[geom_id].is_empty() {
            return;
        }
        mesh_areas[geom_id] = self.meshes[geom_id].areas();
    }
    fn mesh_total_area(&self, geom_id: usize) -> f32 {
        self.compute_mesh_area(geom_id);
        let mesh_areas = self.mesh_areas.borrow();
        mesh_areas[geom_id].par_iter().sum()
    }
    fn estimate_surface_emission_power(
        &self,
        instance: &MeshInstance,
        surface: &ShaderGraph,
    ) -> f32 {
        let out = &surface[&surface.out];
        let out = match out {
            Node::OutputSurface { surface: out } => &surface[out],
            _ => unreachable!(),
        };
        let (emission, strength) = match out {
            Node::PrincipledBsdf {
                emission,
                emission_strength,
                ..
            } => (Some(emission), Some(emission_strength)),
            Node::Emission {
                color: emission,
                strength,
            } => (Some(emission), Some(strength)),
            _ => (None, None),
        };
        let emission = emission
            .map(|e| self.estimate_emission_tex_intensity_fast(&e, surface))
            .flatten();
        let strength = strength
            .map(|s| self.estimate_emission_tex_intensity_fast(&s, surface))
            .flatten();
        if let (Some(power), Some(strength)) = (emission, strength) {
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
        0.0
    }

    fn load_transform(&self, transform: &node::Transform, is_camera: bool) -> AffineTransform {
        match transform {
            node::Transform::TRS(trs) => {
                let coord_sys = trs.coordinate_system;

                let mut m = glam::Mat4::IDENTITY;
                let t = glam::Vec3::from(trs.translation);
                let r = glam::Vec3::from(trs.rotation);
                let s = glam::Vec3::from(trs.scale);
                // dbg!(t,r,s);

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
            node::Transform::Matrix(m) => {
                let m = glam::Mat4::from_cols_array_2d(m).transpose();
                AffineTransform::from_matrix(&m)
            }
        }
    }
    fn load_camera(&self, camera: &node::Camera) -> Arc<dyn Camera> {
        match camera {
            node::Camera::Perspective(cam) => {
                let transform = self.load_transform(&cam.transform, true);
                let fov = cam.fov.to_radians();
                let focal_distance = cam.focal_distance;
                let fstop = cam.fstop;
                let lens_radius = focal_distance / (2.0 * fstop);
                let width = cam.sensor_width;
                let height = cam.sensor_height;
                let cam = Arc::new(PerspectiveCamera::new(
                    self.device.clone(),
                    Uint2::new(width, height),
                    transform,
                    fov,
                    lens_radius,
                    focal_distance,
                ));
                cam
            }
        }
    }
    fn load_instance(&self, instance: &node::Instance) -> (Ref<Material>, MeshInstance) {
        let mat = &instance.material;
        let surface = self.nodes_to_surface_shader[mat];
        let transform = self.load_transform(&instance.transform, false);
        let geometry_node_id = &instance.geometry;
        let geometry_node = &self.graph.geometries[geometry_node_id];
        match geometry_node {
            node::Geometry::Mesh(_) => {
                let geom_id = self.node_to_mesh[geometry_node_id].0;
                let mesh_buffer = &self.mesh_buffers[geom_id];
                (
                    mat.clone(),
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
            }
            _ => todo!(),
        }
    }
    fn do_load(mut self) -> Scene {
        self.mesh_areas
            .borrow_mut()
            .resize(self.mesh_buffers.len(), vec![]);
        for (mat_id, mat) in &self.graph.materials {
            let shader = &mat.shader;
            let shader = self.surface_shader_compiler.compile(SvmCompileContext {
                images: &self.path_sampler_to_idx,
                graph: &shader,
            });
            self.nodes_to_surface_shader.insert(mat_id.clone(), shader);
        }
        // for (light_id, light) in &self.graph.lights {
        //     let surface = &light.in_surface.as_node().unwrap().from;
        //     let shader = self.surface_shader_compiler.compile(
        //         surface,
        //         SvmCompileContext {
        //             images: &self.path_sampler_to_idx,
        //             graph: &self.graph,
        //         },
        //     );
        //     self.nodes_to_surface_shader.insert(surface.clone(), shader);
        // }
        let mut instances = vec![];
        let mut instance_surfaces = vec![];
        let mut instance_nodes = vec![];
        {
            let camera = self.load_camera(&self.graph.camera);
            self.camera = Some(camera);
        }
        for (id, instance_node) in &self.graph.instances {
            let (surface, instance) = self.load_instance(instance_node);
            instances.push(instance);
            instance_surfaces.push(surface);
            instance_nodes.push(id.clone());
        }
        let mut lights = vec![];
        let mut light_ids_to_lights = vec![];
        // now compute light emission power
        for (i, inst) in instances.iter_mut().enumerate() {
            let power = self.estimate_surface_emission_power(
                inst,
                &self.graph.materials[&instance_surfaces[i]].shader,
            );
            if 0.0 < power && power <= 1e-4 {
                log::warn!(
                    "Light power too low: {:?}, power: {}",
                    instance_nodes[i],
                    power
                );
            }
            let instance_node = &self.graph.instances[&instance_nodes[i]];
            if power > 1e-4 {
                let light_id = lights.len();
                lights.push((i, power));

                let geom_id = inst.geom_id as usize;
                self.compute_mesh_area(geom_id);
                let mesh = &mut self.mesh_buffers[geom_id];
                if mesh.area_sampler.is_none() {
                    mesh.build_area_sampler(
                        self.device.clone(),
                        &self.mesh_areas.borrow()[geom_id],
                    );
                }
                let surface_shader = &self.nodes_to_surface_shader[&instance_node.material];
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
        assert!(area_light_count == light_ids_to_lights.len());
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
            use_rq: false,
            // env_map: todo!(),
        }
    }
    fn preload(device: Device, graph: node::Scene) -> Self {
        let mut node_to_mesh = HashMap::new();
        let mut images_to_load: HashSet<String> = HashSet::new();
        let file_resolver = LocalFileResolver::new(vec![]);
        let mut meshes = vec![];
        let mut mesh_buffers = vec![];
        let mut path_samplers: HashSet<(String, TextureSampler)> = HashSet::new();
        // let mut instance_nodes = vec![];
        for tex in &graph.images {
            match &tex.data {
                node::Buffer::External(path) => {
                    let sampler = sampler_from_rgb_image_tex_node(&tex);
                    path_samplers.insert((path.clone(), sampler));
                    images_to_load.insert(path.clone());
                }
                node::Buffer::Internal(_) => todo!(),
            }
        }
        // for id in sorted_node_ids {
        //     let node = &graph.nodes[id];
        //     let ty = node.ty().unwrap();
        //     if ty == nodes::RGBImageTexture::ty() {
        //         let tex = node.proxy::<nodes::RGBImageTexture>(&graph).unwrap();
        //         let path = tex.in_path.as_value().unwrap();
        //         let sampler = sampler_from_rgb_image_tex_node(&tex);
        //         path_samplers.insert((path.clone(), sampler));
        //         images_to_load.insert(path.clone());
        //     } else if ty == nodes::Mesh::ty() {
        for (id, geometry) in &graph.geometries {
            match geometry {
                node::Geometry::Mesh(mesh) => {
                    log::info!("Loading mesh: {}", id.id);
                    let vertices = load_buffer::<[f32; 3]>(&file_resolver, &mesh.vertices);
                    let normals = load_buffer::<[f32; 3]>(&file_resolver, &mesh.normals);
                    let indices = load_buffer::<[u32; 3]>(&file_resolver, &mesh.indices);
                    let uvs = mesh
                        .uvs
                        .as_ref()
                        .map(|b| load_buffer::<[f32; 2]>(&file_resolver, b))
                        .unwrap_or(vec![]);
                    let tangents = mesh
                        .tangents
                        .as_ref()
                        .map(|b| load_buffer::<[f32; 3]>(&file_resolver, b))
                        .unwrap_or(vec![]);
                    let bitangent_signs = mesh
                        .bitangent_signs
                        .as_ref()
                        .map(|b| load_buffer::<u32>(&file_resolver, b))
                        .unwrap_or(vec![]);
                    let mesh = Arc::new(TriangleMesh {
                        name: id.id.clone(),
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
                }
            }
        }
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
                    .unwrap()
                    .flipv();
                (path_s, img)
            })
            .collect::<HashMap<_, _>>();
        log::info!("Loaded {} images", images.len());
        let orded_image_paths = images.keys().collect::<Vec<_>>();
        let textures = {
            orded_image_paths
                .iter()
                .map(|path| {
                    let img = &images[*path];
                    let img = img.to_rgba8();
                    let tex =
                        device.create_tex2d(PixelStorage::Byte4, img.width(), img.height(), 1);
                    let pixels = img.pixels().map(|p| p.0).collect::<Vec<_>>();
                    tex.view(0).copy_from(&pixels);
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
            texture_heap,
            textures,
            surface_shader_compiler: CompilerDriver::new(),
            nodes_to_surface_shader: HashMap::new(),
            mesh_areas: RefCell::new(vec![]),
            lights: PolymorphicBuilder::new(device.clone()),
            path_sampler_to_idx,
        }
    }
}

pub fn load_scene_graph<P: AsRef<Path>>(path: P) -> node::Scene {
    let path = path.as_ref();
    let file = File::open(path).unwrap();
    serde_json::from_reader(file).unwrap()
}

pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Scene> {
    SceneLoader::load_from_path(device, path)
}

pub(crate) fn sampler_from_rgb_image_tex_node(tex: &node::Image) -> TextureSampler {
    let extension = tex.extension;
    let interp = tex.interpolation;
    let sampler = TextureSampler {
        address: match extension {
            node::ImageExtenisionMode::Repeat => SamplerAddress::Repeat,
            node::ImageExtenisionMode::Clip => SamplerAddress::Zero,
            node::ImageExtenisionMode::Mirror => SamplerAddress::Mirror,
            node::ImageExtenisionMode::Extend => SamplerAddress::Edge,
        },
        filter: match interp {
            node::ImageInterpolationMode::Linear => SamplerFilter::LinearLinear,
            node::ImageInterpolationMode::Nearest => SamplerFilter::LinearPoint,
            node::ImageInterpolationMode::Cubic => {
                log::warn!(
                    "Cubic interpolation is not supported, falling back to linear interpolation"
                );
                SamplerFilter::LinearLinear
            }
        },
    };
    sampler
}
