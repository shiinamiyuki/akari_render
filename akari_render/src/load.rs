use crate::{
    camera::{Camera, PerspectiveCamera},
    geometry::AffineTransform,
    heap::MegaHeap,
    light::{area::AreaLight, Light, LightAggregate, WeightedLightDistribution},
    mesh::{Mesh, MeshAggregate, MeshBuildArgs, MeshInstanceHost},
    scene::Scene,
    svm::{
        compiler::{CompilerDriver, SvmCompileContext},
        ShaderRef, Svm,
    },
    *,
};
use akari_scenegraph as scenegraph;
use akari_scenegraph::{CoordinateSystem, Geometry, Material, NodeRef, ShaderGraph, ShaderNode};
use image::io::Reader as ImageReader;
use luisa::runtime::api::denoiser_ext::Image;

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    f32::consts::PI,
    fs::File,
    path::Path,
    sync::Arc,
};
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ImageKey {
    buffer: NodeRef<scenegraph::Buffer>,
    format: scenegraph::ImageFormat,
    extension: scenegraph::ImageExtenisionMode,
    interpolation: scenegraph::ImageInterpolationMode,
    width: u32,
    height: u32,
    channels: u32,
}
pub struct SceneLoader {
    device: Device,
    #[allow(dead_code)]
    images: HashMap<ImageKey, Arc<Tex2d<Float4>>>,
    #[allow(dead_code)]
    textures: Vec<Arc<Tex2d<Float4>>>,
    node_to_geom_id: HashMap<NodeRef<Geometry>, usize>,
    mesh_areas: RefCell<Vec<Vec<f32>>>,
    mesh_buffers: Vec<Mesh>,
    graph: scenegraph::Scene,
    camera: Option<Arc<dyn Camera>>,
    surface_shader_compiler: CompilerDriver,
    lights: PolymorphicBuilder<(), dyn Light>,
    nodes_to_surface_shader: HashMap<NodeRef<Material>, ShaderRef>,
    image_key_sampler_to_idx: HashMap<(ImageKey, TextureSampler), usize>,
    heap: Arc<MegaHeap>,
}

#[deny(dead_code)]
impl SceneLoader {
    pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Scene> {
        let abs_path = path.as_ref().canonicalize().unwrap();
        log::info!("Loading scene: {}", abs_path.display());
        let mut graph = load_scene_graph(&path);
        log::info!("Loaded scene graph");
        let parent_path = abs_path.parent().unwrap();
        graph.embed(parent_path).unwrap();
        let loader = Self::preload(device, graph);
        let scene = Arc::new(loader.do_load());
        log::info!("Loaded scene: {}", abs_path.display());
        scene
    }
    fn estimate_emission_tex_intensity_fast(
        &self,
        emission: &NodeRef<ShaderNode>,
        shader_graph: &ShaderGraph,
    ) -> Option<f32> {
        let emission = &shader_graph[emission];
        match emission {
            ShaderNode::SpectralUplift { rgb } => {
                self.estimate_emission_tex_intensity_fast(rgb, shader_graph)
            }
            ShaderNode::Float { value: x } => return Some(*x),
            ShaderNode::Float3 { value: x } => {
                return Some(x.iter().copied().reduce(|a, b| a.max(b)).unwrap())
            }
            ShaderNode::Rgb { value: rgb, .. } => {
                Some(rgb.iter().copied().reduce(|a, b| a.max(b)).unwrap())
            }
            _ => return None,
        }
    }
    fn compute_mesh_area(&self, geom_id: usize) {
        todo!("this is incorrect, need to accound for transform")
        // let mut mesh_areas = self.mesh_areas.borrow_mut();
        // if !mesh_areas[geom_id].is_empty() {
        //     return;
        // }
        // mesh_areas[geom_id] = self.meshes[geom_id].areas();
    }
    fn mesh_total_area(&self, geom_id: usize) -> f32 {
        self.compute_mesh_area(geom_id);
        let mesh_areas = self.mesh_areas.borrow();
        mesh_areas[geom_id].par_iter().sum()
    }
    fn estimate_surface_emission_power(
        &self,
        instance: &MeshInstanceHost,
        surface: &ShaderGraph,
    ) -> f32 {
        let out = &surface[&surface.output];
        let out = match out {
            ShaderNode::Output { node: out } => &surface[out],
            _ => unreachable!(),
        };
        let (emission, strength) = match out {
            ShaderNode::PrincipledBsdf {
                emission,
                emission_strength,
                ..
            } => (Some(emission), Some(emission_strength)),
            ShaderNode::Emission {
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
            let transform: glam::Mat3 =
                glam::Mat3::from_mat4(glam::Mat4::from(instance.transform.m));
            let det = transform.determinant();
            // dbg!(power, area, det, strength);
            return power * area * det.abs() * strength;
        }
        0.0
    }

    fn load_transform(
        &self,
        transform: &scenegraph::Transform,
        is_camera: bool,
    ) -> AffineTransform {
        match transform {
            scenegraph::Transform::TRS(trs) => {
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
            scenegraph::Transform::Matrix(m) => {
                let m = glam::Mat4::from_cols_array_2d(m).transpose();
                AffineTransform::from_matrix(&m)
            }
        }
    }
    fn load_camera(&self, camera: &scenegraph::Camera) -> Arc<dyn Camera> {
        match camera {
            scenegraph::Camera::Perspective(cam) => {
                let transform = self.load_transform(&cam.transform, true);
                let fov = cam.fov.to_radians();
                let focal_distance = cam.focal_distance;
                let fstop = cam.fstop;
                let lens_radius = focal_distance / (2.0 * fstop);
                let width = cam.sensor_width;
                let height = cam.sensor_height;
                let cam = Arc::new(PerspectiveCamera::new(
                    self.device.clone(),
                    &self.heap,
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
    fn load_instance(
        &self,
        instance: &scenegraph::Instance,
    ) -> (NodeRef<Material>, MeshInstanceHost) {
        let mat = &instance.material;
        let surface = self.nodes_to_surface_shader[mat];
        let transform = self.load_transform(&instance.transform, false);
        let geometry_node_id = &instance.geometry;
        let geometry_node = &self.graph.geometries[geometry_node_id];
        match geometry_node {
            scenegraph::Geometry::Mesh(_) => {
                let geom_id = self.node_to_geom_id[geometry_node_id];
                let mesh_buffer = &self.mesh_buffers[geom_id];
                (
                    mat.clone(),
                    MeshInstanceHost {
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
                images: &self.image_key_sampler_to_idx,
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
                        geom_id: geom_id as u32,
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
            self.mesh_buffers.len(),
            instances.len()
        );
        let mesh_aggregate = Arc::new(MeshAggregate::new(
            self.device.clone(),
            &self.heap,
            &self.mesh_buffers.iter().collect::<Vec<_>>(),
            &instances,
        ));
        let light_weights = lights.iter().map(|(_, power)| *power).collect::<Vec<_>>();
        let Self {
            device,
            lights,
            surface_shader_compiler,
            camera,
            heap,
            ..
        } = self;
        let svm = Arc::new(Svm {
            device: device.clone(),
            surface_shaders: surface_shader_compiler.upload(&device),
            heap: heap.clone(),
        });
        log::info!(
            "Shader variant count: {}",
            svm.surface_shaders.variant_count()
        );
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
        heap.commit();
        Scene {
            svm,
            lights: light_aggregate,
            meshes: mesh_aggregate.clone(),
            camera: camera.unwrap(),
            device,
            use_rq: true,
            // env_map: todo!(),
            heap,
        }
    }
    fn preload(device: Device, graph: scenegraph::Scene) -> Self {
        let mut node_to_geom_id = HashMap::new();
        let mut images_to_load: HashSet<ImageKey> = HashSet::new();
        let mut meshes = vec![];
        let mut texture_and_sampler_pair: HashSet<(ImageKey, TextureSampler)> = HashSet::new();
        // let mut instance_nodes = vec![];

        let image_nodes = {
            let mut images = vec![];
            for (_, mat) in &graph.materials {
                for (_, n) in &mat.shader.nodes {
                    match n {
                        ShaderNode::TexImage { image: tex } => images.push(tex.clone()),
                        _ => {}
                    }
                }
            }
            images
        };

        for tex in &image_nodes {
            let buf = tex.data.clone();
            let sampler = sampler_from_rgb_image_tex_node(tex);

            let key = ImageKey {
                buffer: buf,
                interpolation: tex.interpolation,
                extension: tex.extension,
                format: tex.format,
                width: tex.width,
                height: tex.height,
                channels: tex.channels,
            };
            images_to_load.insert(key.clone());
            texture_and_sampler_pair.insert((key, sampler));
        }

        for (id, geometry) in &graph.geometries {
            match geometry {
                scenegraph::Geometry::Mesh(mesh) => {
                    log::debug!("Loading mesh: {}", id.id);
                    let vertices = graph.buffers[&mesh.vertices].as_slice::<[f32; 3]>();
                    let normals = mesh
                        .normals
                        .as_ref()
                        .map(|b| graph.buffers[b].as_slice::<[f32; 3]>());
                    let indices = graph.buffers[&mesh.indices].as_slice::<[u32; 3]>();
                    let uvs = mesh
                        .uvs
                        .as_ref()
                        .map(|b| graph.buffers[b].as_slice::<[f32; 2]>());
                    let tangents = mesh
                        .tangents
                        .as_ref()
                        .map(|b| graph.buffers[b].as_slice::<[f32; 3]>());
                    // let bitangent_signs: Vec<u32> = mesh
                    //     .bitangent_signs
                    //     .as_ref()
                    //     .map(|b| load_buffer::<u32>(&file_resolver, b))
                    //     .unwrap_or(vec![]);
                    let mesh = Mesh::new(
                        device.clone(),
                        MeshBuildArgs {
                            vertices,
                            normals,
                            indices,
                            uvs,
                            tangents,
                        },
                    );
                    let geom_id = meshes.len();
                    meshes.push(mesh);
                    node_to_geom_id.insert(id.clone(), geom_id);
                }
            }
        }
        log::info!(
            "Scene has {} meshes, {} image textures",
            meshes.len(),
            images_to_load.len()
        );
        let images_to_load = images_to_load.into_iter().collect::<Vec<_>>();
        let images: HashMap<ImageKey, Arc<Tex2d<Float4>>> = images_to_load
            .into_par_iter()
            .map(|key| {
                let buf = &graph.buffers[&key.buffer];
                let data = buf.as_binary_data();

                assert!(
                    key.channels <= 4,
                    "Invalid number of channels: {}",
                    key.channels
                );
                let tex = if key.format == scenegraph::ImageFormat::Float {
                    assert_eq!(
                        data.len(),
                        key.width as usize * key.height as usize * key.channels as usize * 4
                    );
                    let mut rgbaf32 = vec![];
                    let img_data = if key.channels != 4 {
                        let data = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                        };
                        rgbaf32.reserve(data.len() * 4);
                        for i in 0..(data.len() / key.channels as usize) {
                            for c in 0..key.channels {
                                rgbaf32.push(data[i * key.channels as usize + c as usize]);
                            }
                            for c in key.channels..4 {
                                rgbaf32.push(0.0);
                            }
                        }
                        rgbaf32.as_slice()
                    } else {
                        unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                        }
                    };
                    let tex = device.create_tex2d::<Float4>(
                        PixelStorage::Float4,
                        key.width,
                        key.height,
                        1,
                    );
                    tex.view(0).copy_from(img_data);
                    tex
                } else {
                    let cursor = std::io::Cursor::new(data);
                    let format = match key.format {
                        scenegraph::ImageFormat::Png => image::ImageFormat::Png,
                        scenegraph::ImageFormat::Jpeg => image::ImageFormat::Jpeg,
                        scenegraph::ImageFormat::Tiff => image::ImageFormat::Tiff,
                        scenegraph::ImageFormat::OpenExr => image::ImageFormat::OpenExr,
                        _ => unreachable!(),
                    };
                    let mut reader = ImageReader::new(cursor);
                    reader.set_format(format);
                    let img = reader.decode().unwrap().flipv();
                    if key.format != scenegraph::ImageFormat::OpenExr {
                        let img = img.to_rgba8();
                        let tex =
                            device.create_tex2d(PixelStorage::Byte4, img.width(), img.height(), 1);
                        let pixels = img.pixels().map(|p| p.0).collect::<Vec<_>>();
                        tex.view(0).copy_from(&pixels);
                        tex
                    } else {
                        let img = img.to_rgba32f();
                        let tex =
                            device.create_tex2d(PixelStorage::Float4, img.width(), img.height(), 1);
                        let pixels = img.pixels().map(|p| p.0).collect::<Vec<_>>();
                        tex.view(0).copy_from(&pixels);
                        tex
                    }
                };
                (key, Arc::new(tex))
            })
            .collect::<HashMap<_, _>>();
        log::info!("Loaded {} images", images.len());
        let mut ordered_image_paths = images.keys().collect::<Vec<_>>();
        ordered_image_paths.sort_by(|a, b| a.buffer.cmp(&b.buffer));
        let textures = {
            ordered_image_paths
                .iter()
                .map(|path| images[path].clone())
                .collect::<Vec<_>>()
        };
        let heap = MegaHeap::new(device.clone(), 131072);
        let path_to_texture_idx = ordered_image_paths
            .into_iter()
            .enumerate()
            .map(|(i, path)| (path.clone(), i))
            .collect::<HashMap<ImageKey, usize>>();
        let image_key_sampler_to_idx = {
            let mut image_key_sampler_to_idx = HashMap::new();
            for (path, sampler) in &texture_and_sampler_pair {
                let key: (ImageKey, TextureSampler) = (path.clone(), sampler.clone());
                if let None = image_key_sampler_to_idx.get(&key) {
                    let idx = path_to_texture_idx[path];
                    let idx = heap.bind_tex2d(&textures[idx], *sampler);
                    image_key_sampler_to_idx.insert(key, idx as usize);
                }
            }

            image_key_sampler_to_idx
        };
        // heap.commit();
        log::info!(
            "Texture Heap contains total {} (texture, sampler) tuples",
            image_key_sampler_to_idx.len()
        );
        Self {
            device: device.clone(),
            images,
            node_to_geom_id,
            mesh_buffers: meshes,
            graph,
            camera: None,
            heap: Arc::new(heap),
            textures,
            surface_shader_compiler: CompilerDriver::new(),
            nodes_to_surface_shader: HashMap::new(),
            mesh_areas: RefCell::new(vec![]),
            lights: PolymorphicBuilder::new(device.clone()),
            image_key_sampler_to_idx,
        }
    }
}

pub fn load_scene_graph<P: AsRef<Path>>(path: P) -> scenegraph::Scene {
    let path = path.as_ref();
    let file = File::open(path).unwrap();
    serde_json::from_reader(file).unwrap()
}

pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Scene> {
    SceneLoader::load_from_path(device, path)
}

pub(crate) fn sampler_from_rgb_image_tex_node(tex: &scenegraph::Image) -> TextureSampler {
    let extension = tex.extension;
    let interp = tex.interpolation;
    let sampler = TextureSampler {
        address: match extension {
            scenegraph::ImageExtenisionMode::Repeat => SamplerAddress::Repeat,
            scenegraph::ImageExtenisionMode::Clip => SamplerAddress::Zero,
            scenegraph::ImageExtenisionMode::Mirror => SamplerAddress::Mirror,
            scenegraph::ImageExtenisionMode::Extend => SamplerAddress::Edge,
        },
        filter: match interp {
            scenegraph::ImageInterpolationMode::Linear => SamplerFilter::LinearLinear,
            scenegraph::ImageInterpolationMode::Nearest => SamplerFilter::LinearPoint,
            scenegraph::ImageInterpolationMode::Cubic => {
                log::warn!(
                    "Cubic interpolation is not supported, falling back to linear interpolation"
                );
                SamplerFilter::LinearLinear
            }
        },
    };
    sampler
}
