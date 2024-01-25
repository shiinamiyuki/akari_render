use crate::{
    camera::{Camera, PerspectiveCamera},
    color::{sample_wavelengths, ColorPipeline},
    geometry::AffineTransform,
    heap::MegaHeap,
    light::{area::AreaLight, Light, LightAggregate, WeightedLightDistribution},
    mesh::{Mesh, MeshAggregate, MeshInstanceFlags, MeshInstanceHost, MeshRef},
    sampler::{IndependentSampler, Pcg32Var, Sampler},
    sampling::{cos_sample_hemisphere, uniform_sample_triangle},
    scene::Scene,
    svm::{
        compiler::{CompilerDriver, SvmCompileContext},
        surface::{BsdfEvalContext, PreComputedTables, Surface},
        ShaderRef, Svm,
    },
    util::distribution::AliasTable,
    *,
};
use akari_common::parking_lot::lock_api::Mutex;
use akari_scenegraph as scenegraph;
use akari_scenegraph::{CoordinateSystem, Geometry, Material, NodeRef, ShaderGraph, ShaderNode};
use image::io::Reader as ImageReader;
use scene_graph::{MmapScene, SceneView};

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
    pub buffer: NodeRef<scenegraph::BufferView>,
    pub format: scenegraph::ImageFormat,
    pub extension: scenegraph::ImageExtenisionMode,
    pub interpolation: scenegraph::ImageInterpolationMode,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
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
    scene_view: Box<dyn SceneView + Sync>,
    camera: Option<Arc<dyn Camera>>,
    surface_shader_compiler: Option<CompilerDriver>,
    lights: PolymorphicBuilder<(), dyn Light>,
    nodes_to_surface_shader: HashMap<NodeRef<Material>, ShaderRef>,
    image_key_sampler_to_idx: HashMap<(ImageKey, TextureSampler), usize>,
    heap: Arc<MegaHeap>,
}

// #[deny(dead_code)]
impl SceneLoader {
    pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Scene> {
        let abs_path = path.as_ref().canonicalize().unwrap();
        log::info!("Loading scene: {}", abs_path.display());
        let scene_view = MmapScene::open(&abs_path).unwrap();
        log::info!("Loaded scene graph");
        let loader = Self::preload(device, Box::new(scene_view));
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
    fn has_potential_surface_emission(
        &self,
        instance: &MeshInstanceHost,
        surface: &ShaderGraph,
    ) -> bool {
        let out = &surface[&surface.output];
        let out = match out {
            ShaderNode::Output { node: out } => &surface[out],
            _ => unreachable!(),
        };
        let (emission, strength) = match out {
            ShaderNode::PrincipledBsdf { bsdf } => {
                (Some(&bsdf.emission_color), Some(&bsdf.emission_strength))
            }
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
            if power * strength == 0.0 {
                return false;
            }
            return true;
        }
        true
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
    ) -> (Vec<NodeRef<Material>>, MeshInstanceHost) {
        let mat = &instance.materials;
        let surfaces = mat
            .iter()
            .map(|mat| self.nodes_to_surface_shader[mat])
            .collect::<Vec<_>>();
        let transform = self.load_transform(&instance.transform, false);
        let geometry_node_id = &instance.geometry;
        let geometry_node = &self.scene_view.scene().geometries[geometry_node_id];
        match geometry_node {
            scenegraph::Geometry::Mesh(_) => {
                let geom_id = self.node_to_geom_id[geometry_node_id];
                let mesh_buffer = &self.mesh_buffers[geom_id];
                let mut flags: u32 = 0;
                if mesh_buffer.has_normals {
                    flags |= MeshInstanceFlags::HAS_NORMALS;
                }
                if mesh_buffer.has_uvs {
                    flags |= MeshInstanceFlags::HAS_UVS;
                }
                if mesh_buffer.has_tangents {
                    flags |= MeshInstanceFlags::HAS_TANGENTS;
                }
                if mesh_buffer.material_slots.len() > 1 {
                    flags |= MeshInstanceFlags::HAS_MULTI_MATERIALS;
                }
                (
                    mat.clone(),
                    MeshInstanceHost {
                        geom_id: geom_id as u32,
                        transform,
                        light: TagIndex::INVALID,
                        materials: surfaces,
                        flags,
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
        for (mat_id, mat) in &self.scene_view.scene().materials {
            let shader = &mat.shader;
            let shader =
                self.surface_shader_compiler
                    .as_mut()
                    .unwrap()
                    .compile(SvmCompileContext {
                        images: &self.image_key_sampler_to_idx,
                        graph: &shader,
                    });
            self.nodes_to_surface_shader.insert(mat_id.clone(), shader);
        }
        let svm = Arc::new(Svm {
            device: self.device.clone(),
            surface_shaders: self
                .surface_shader_compiler
                .take()
                .unwrap()
                .upload(&self.device),
            heap: self.heap.clone(),
            precompute_tables: Mutex::new(HashMap::new()),
        });
        svm.init_precompute_tables(ColorRepr::Rgb(color::RgbColorSpace::SRgb));
        log::info!(
            "Shader variant count: {}",
            svm.surface_shaders().variant_count()
        );
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
            let camera = self.load_camera(self.scene_view.scene().camera.as_ref().unwrap());
            self.camera = Some(camera);
        }
        for (id, instance_node) in &self.scene_view.scene().instances {
            let (surface, instance) = self.load_instance(instance_node);
            instances.push(instance);
            instance_surfaces.push(surface);
            instance_nodes.push(id.clone());
        }

        log::info!(
            "Building accel for {} meshes, {} instances",
            self.mesh_buffers.len(),
            instances.len()
        );
        self.heap.commit();
        let mut mesh_aggregate = MeshAggregate::new(
            self.device.clone(),
            &self.heap,
            &self.mesh_buffers.iter().collect::<Vec<_>>(),
            &instances,
        );
        self.heap.commit();

        let mut lights = vec![];
        let mut light_ids_to_lights = vec![];
        let mut powers = self.device.create_buffer::<f32>(1024);

        let esimate_emisson_kernel = self.device.create_kernel::<fn(u32, Buffer<f32>)>(&track!(
            |inst_id: Expr<u32>, powers: BufferVar<f32>| {
                let i = dispatch_id().x;
                let rng = Pcg32Var::new_seq(i.as_u64());
                let color_repr = ColorRepr::Rgb(color::RgbColorSpace::SRgb);
                let color_pipeline = ColorPipeline {
                    color_repr,
                    rgb_colorspace: color::RgbColorSpace::SRgb,
                };
                let sampler = IndependentSampler::from_pcg32(rng);
                let acc = 0.0f32.var();
                let n_samples = 16;
                for _ in 0..n_samples {
                    let bary = uniform_sample_triangle(sampler.next_2d());
                    let swl = sample_wavelengths(color_repr, &sampler).var();
                    let si = mesh_aggregate.surface_interaction(inst_id, i, bary);
                    let bsdf_eval_ctx = BsdfEvalContext {
                        color_repr,
                        ad_mode: ADMode::None,
                    };
                    let onb = si.frame;
                    let wo = cos_sample_hemisphere(sampler.next_2d());
                    let wo = onb.to_world(wo);
                    svm.dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                        let emission = closure.emission(wo, **swl, &bsdf_eval_ctx);
                        // device_log!("surface: {}, prim {} emission: {}", si.surface, i, emission.max());
                        *acc += emission.max() * si.prim_area;
                    })
                }
                powers.write(i, acc / n_samples as f32);
            }
        ));
        // now compute light emission power
        for (i, inst) in instances.iter_mut().enumerate() {
            let has_potential_emission = instance_surfaces[i]
                .iter()
                .map(|s| {
                    self.has_potential_surface_emission(
                        inst,
                        &self.scene_view.scene().materials[s].shader,
                    )
                })
                .collect::<Vec<_>>();
            if !has_potential_emission.iter().any(|&b| b) {
                continue;
            }
            let geom_id = inst.geom_id as usize;
            let powers = {
                let mesh = &self.mesh_buffers[geom_id];
                if powers.len() < mesh.indices.len() {
                    powers = self.device.create_buffer::<f32>(mesh.indices.len());
                }
                esimate_emisson_kernel.dispatch(
                    [mesh.indices.len() as u32, 1, 1],
                    &(i as u32),
                    &powers,
                );
                powers.view(..mesh.indices.len()).copy_to_vec()
            };
            let total_power = powers
                .iter()
                .enumerate()
                .map(|(j, x)| {
                    if !x.is_finite() {
                        log::error!(
                            "non-finite power detected at instance {}, prim {}",
                            &instance_nodes[i],
                            j
                        );
                    }
                    x
                })
                .sum::<f32>();
            if 0.0 < total_power && total_power <= 1e-4 {
                log::warn!(
                    "Light power too low: {:?}, power: {}",
                    instance_nodes[i],
                    total_power
                );
            }

            if total_power > 1e-4 {
                let light_id = lights.len();
                lights.push((i, total_power));
                log::info!(
                    "Detected mesh light @ Instance {} with power {}",
                    &instance_nodes[i],
                    total_power
                );
                // dbg!(&powers);
                let at = AliasTable::new(self.device.clone(), &powers);
                mesh_aggregate.set_area_sampler(i as u32, at);
                let light_ref = self.lights.push(
                    (),
                    AreaLight {
                        light_id: light_id as u32,
                        instance_id: i as u32,
                        geom_id: geom_id as u32,
                    },
                );
                light_ids_to_lights.push(light_ref);
                mesh_aggregate.set_instance_light(i as u32, light_ref);
            }
        }
        mesh_aggregate.commit();
        let area_light_count = lights.len();
        assert!(area_light_count == light_ids_to_lights.len());
        log::info!("{} mesh lights found", area_light_count);
        // now add other lights
        {
            // TODO: add other lights
        }
        let mesh_aggregate = Arc::new(mesh_aggregate);
        let light_weights = lights.iter().map(|(_, power)| *power).collect::<Vec<_>>();
        let Self {
            device,
            lights,
            camera,
            heap,
            ..
        } = self;

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
    fn preload(device: Device, scene_view: Box<dyn SceneView + Sync>) -> Self {
        let mut node_to_geom_id = HashMap::new();
        let mut images_to_load: HashSet<ImageKey> = HashSet::new();
        let mut meshes = vec![];
        let mut texture_and_sampler_pair: HashSet<(ImageKey, TextureSampler)> = HashSet::new();
        // let mut instance_nodes = vec![];

        let image_nodes = {
            let mut images = vec![];
            for (_, mat) in &scene_view.scene().materials {
                for (_, n) in &mat.shader.nodes {
                    match n {
                        ShaderNode::TexImage { image: tex, .. } => images.push(tex.clone()),
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

        for (id, geometry) in &scene_view.scene().geometries {
            match geometry {
                scenegraph::Geometry::Mesh(mesh) => unsafe {
                    macro_rules! load_slice {
                        ($s:expr, $t:ty) => {{
                            let slice = scene_view.buffer_view_as_slice($s);
                            assert_eq!(
                                slice.len() % std::mem::size_of::<$t>(),
                                0,
                                "Invalid slice length"
                            );
                            let slice = std::slice::from_raw_parts(
                                slice.as_ptr() as *const $t,
                                slice.len() / std::mem::size_of::<$t>(),
                            );
                            slice
                        }};
                    }
                    let vertices = load_slice!(&mesh.vertices, [f32; 3]);
                    let normals = mesh.normals.as_ref().map(|n| load_slice!(n, [f32; 3]));
                    let indices = load_slice!(&mesh.indices, [u32; 3]);
                    let uvs = mesh.uvs.as_ref().map(|uvs| load_slice!(uvs, [f32; 2]));
                    let tangents = mesh.tangents.as_ref().map(|t| load_slice!(t, [f32; 3]));
                    let materials = load_slice!(&mesh.materials, u32);
                    // if materials.len() > 1 {
                    //     dbg!(materials);
                    // }
                    let mesh = Mesh::new(
                        device.clone(),
                        MeshRef::new(vertices, normals, indices, materials, uvs, tangents),
                    );
                    let geom_id = meshes.len();
                    meshes.push(mesh);
                    node_to_geom_id.insert(id.clone(), geom_id);
                },
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
                let data = scene_view.buffer_view_as_slice(&key.buffer);

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
                                if c != 3 {
                                    rgbaf32.push(0.0);
                                } else {
                                    rgbaf32.push(1.0);
                                }
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
                    let mut reader = ImageReader::new(cursor);

                    let format = match key.format {
                        scenegraph::ImageFormat::Png => image::ImageFormat::Png,
                        scenegraph::ImageFormat::Jpeg => image::ImageFormat::Jpeg,
                        scenegraph::ImageFormat::Tiff => image::ImageFormat::Tiff,
                        scenegraph::ImageFormat::OpenExr => image::ImageFormat::OpenExr,
                        scenegraph::ImageFormat::Dds => image::ImageFormat::Dds,
                        _ => unreachable!(),
                    };

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
            scene_view,
            camera: None,
            heap: Arc::new(heap),
            textures,
            surface_shader_compiler: Some(CompilerDriver::new()),
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
