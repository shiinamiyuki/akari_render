use super::soa::{MaterialEvalInfo, SOAPathState, SOARay, SOAVec};
use crate::camera::{Camera, PerspectiveCamera};
use crate::film::Film;

use crate::gpu::scene::GPUScene;
use crate::integrator::Integrator;
use crate::scene::Scene;
use crate::*;
use ash::vk;
use vkc::include_spv;
use vkc::profile::Profiler;
use vkc::resource::TBuffer;
use vkc::Context;
pub struct WavefrontPathTracer {
    pub max_depth: u32,
    pub spp: u32,
    pub training_iters: u32,
}

impl WavefrontPathTracer {
    pub fn render(&mut self, gpu_scene: &GPUScene, scene: &Scene) -> Film {
        let film = Film::new(&scene.camera.resolution());
        let imp = WavefrontImpl::new(
            &gpu_scene.ctx,
            &*self,
            scene,
            gpu_scene,
            self.max_depth,
            self.spp,
            self.training_iters,
        );
        log::info!("Integrator: {}", "Wavefront Path Tracer");
        imp.render();
        {
            let mapped = imp.film.map_range(.., vk::MemoryMapFlags::empty());
            let mut pixels = film.pixels.write().unwrap();
            assert!(pixels.len() == mapped.slice.len());
            for i in 0..mapped.slice.len() {
                pixels[i].intensity[0] = mapped.slice[i][0];
                pixels[i].intensity[1] = mapped.slice[i][1];
                pixels[i].intensity[2] = mapped.slice[i][2];
                pixels[i].weight = mapped.slice[i][3];
            }
        }
        film
    }
}

struct ClosestHitWorkQueue {
    ray: SOARay,
    sid: TBuffer<i32>,
}

struct ShadowRayWorkQueue {
    ray: SOARay,
    ld: SOAVec<f32, 3>,
    sid: TBuffer<i32>,
}
impl ShadowRayWorkQueue {
    fn new(ctx: &Context, size: usize) -> Self {
        let ray = SOARay::new(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let sid = TBuffer::new(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let ld = SOAVec::<f32, 3>::new(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self { ray, sid, ld }
    }
}

struct InvocationContext<'a> {
    scanlines_per_pass: u32,
    cur_scanline: u32,
    command_encoder: &'a vkc::CommandEncoder<'a>,
    pixels_per_scanline: u32,
    total_scnalines: u32,
}
trait CameraRayGen {
    fn gen<'a>(&self, invoke: &InvocationContext<'a>);
}

impl ClosestHitWorkQueue {
    fn new(ctx: &Context, size: usize, descriptor_pool: vk::DescriptorPool) -> Self {
        let ray = SOARay::new(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let sid = TBuffer::new(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self { ray, sid }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GPUTransform {
    m4: [[f32; 4]; 4],
    inv_m4: [[f32; 4]; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GPUPerspectiveCamera {
    c2w: GPUTransform,
    r2c: GPUTransform,
    resolution: [u32; 2],
}
#[allow(dead_code)]
struct GenericCameraRayGen<T: bytemuck::Pod> {
    ctx: Context,
    uniform: TBuffer<T>,
    kernel: vkc::ComputeKernel,
    ray_queue_buffers: vkc::Set,
    scene_set: vkc::Set,
    // pipeline: vk::Pipeline,
    // pipeline_layout: vk::PipelineLayout,
    // descriptor_set_layout: vk::DescriptorSetLayout,
    // shader_module: vk::ShaderModule,
    // uniform_descriptor_set: vk::DescriptorSet,
    // ray_queue_descriptor_set: vk::DescriptorSet,
    // scene_set: vk::DescriptorSet,
}

impl<T: bytemuck::Pod> GenericCameraRayGen<T> {
    fn new(
        ctx: &Context,
        descriptor_pool: vk::DescriptorPool,
        gpu_scene: &GPUScene,
        ray_queue: &ClosestHitWorkQueue,
        scene_set: vkc::Set,
        spirv: &[u32],
        data: T,
    ) -> Self {
        let uniform = TBuffer::new(
            ctx,
            1,
            vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        {
            let mapped = uniform.map_range_mut(.., vk::MemoryMapFlags::empty());
            mapped.slice[0] = data;
        }
        let kernel = vkc::ComputeKernel::new(ctx, spirv);
        let ray_queue_buffers = vkc::Set::Bindings(vec![
            vkc::Binding::StorageBuffer(ray_queue.ray.o.buffers[0].handle),
            vkc::Binding::StorageBuffer(ray_queue.ray.o.buffers[1].handle),
            vkc::Binding::StorageBuffer(ray_queue.ray.o.buffers[2].handle),
            vkc::Binding::StorageBuffer(ray_queue.ray.d.buffers[0].handle),
            vkc::Binding::StorageBuffer(ray_queue.ray.d.buffers[1].handle),
            vkc::Binding::StorageBuffer(ray_queue.ray.d.buffers[2].handle),
            vkc::Binding::StorageBuffer(ray_queue.ray.tmin.handle),
            vkc::Binding::StorageBuffer(ray_queue.ray.tmax.handle),
            vkc::Binding::StorageBuffer(ray_queue.sid.handle),
        ]);
        Self {
            ctx: ctx.clone(),
            kernel,
            uniform,
            ray_queue_buffers,
            scene_set,
        }
    }
}
impl<T: bytemuck::Pod> CameraRayGen for GenericCameraRayGen<T> {
    fn gen<'a>(&self, invoke: &InvocationContext<'a>) {
        let global_size = invoke.pixels_per_scanline * invoke.scanlines_per_pass;
        self.kernel.cmd_dispatch(
            &invoke.command_encoder,
            (global_size + 255) / 256,
            1,
            1,
            &vkc::KernelArgs {
                sets: vec![
                    vkc::Set::Bindings(vec![vkc::Binding::UniformBuffer(self.uniform.handle)]),
                    self.ray_queue_buffers.clone(),
                    self.scene_set.clone(),
                ],
                push_constants: Some([
                    invoke.cur_scanline,
                    invoke.scanlines_per_pass,
                    invoke.pixels_per_scanline,
                    invoke.total_scnalines,
                ]),
            },
        );
    }
}

#[allow(dead_code)]
struct WavefrontImpl<'a> {
    descriptor_pool: vk::DescriptorPool,
    command_buffer: vk::CommandBuffer,
    sampler: vkc::resource::Sampler,
    film: TBuffer<[f32; 4]>,
    scene_set: vkc::Set,

    wq_counters: TBuffer<u32>,
    path_states: SOAPathState,
    material_eval_info: MaterialEvalInfo,
    rchit_wq: Vec<ClosestHitWorkQueue>,
    shadow_ray_wq: ShadowRayWorkQueue,
    camera_rgen: Box<dyn CameraRayGen>,
    pixels_per_scanline: u32,
    scanlines_per_pass: u32,
    spp: u32,
    gpu_scene: &'a GPUScene,
    scene: &'a Scene,
    ctx: Context,
    init_path_states_kernel: vkc::Kernel,
    intersect_kernel: vkc::Kernel,
    shadow_ray_kernel: vkc::Kernel,
    reset_queue_kernel: vkc::ComputeKernel,
    reset_ray_queue_kernel: vkc::ComputeKernel,
    material_eval_kernel: vkc::Kernel,
    splat_film_kernel: vkc::Kernel,
    max_depth: u32,
    training_iter: u32,
}
impl<'a> Drop for WavefrontImpl<'a> {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_descriptor_pool(
                self.descriptor_pool,
                self.ctx.allocation_callbacks.as_ref(),
            );
        }
    }
}
impl<'a> WavefrontImpl<'a> {
    fn new(
        ctx: &Context,
        pt: &WavefrontPathTracer,
        scene: &'a Scene,
        gpu_scene: &'a GPUScene,
        max_depth: u32,
        spp: u32,
        training_iter: u32,
    ) -> Self {
        unsafe {
            let descriptor_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
                    descriptor_count: 3,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 128,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 128,
                },
            ];
            let pixels_per_scanline = scene.camera.resolution().x;
            let npixels = pixels_per_scanline * scene.camera.resolution().y;
            let block_size = 512 * 512;
            let scanlines_per_pass = std::cmp::max(block_size / pixels_per_scanline, 1);
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(ctx.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .build();

            let command_buffer = ctx.device.allocate_command_buffers(&allocate_info).unwrap()[0];

            let descriptor_pool = ctx
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .pool_sizes(&descriptor_sizes)
                        .max_sets(64),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();
            let shadow_ray_wq =
                ShadowRayWorkQueue::new(ctx, (scanlines_per_pass * pixels_per_scanline) as usize);
            let rchit_wq = vec![
                ClosestHitWorkQueue::new(
                    ctx,
                    (scanlines_per_pass * pixels_per_scanline) as usize,
                    descriptor_pool,
                ),
                ClosestHitWorkQueue::new(
                    ctx,
                    (scanlines_per_pass * pixels_per_scanline) as usize,
                    descriptor_pool,
                ),
            ];
            let ray_queue = &rchit_wq[0];
            let sampler = vkc::resource::Sampler::new(ctx);
            let scene_set = vkc::Set::Bindings(vec![
                vkc::Binding::StorageBuffer(gpu_scene.instances.handle),
                vkc::Binding::StorageBuffer(gpu_scene.bsdfs.handle),
                vkc::Binding::Sampler(sampler.handle),
                vkc::Binding::StorageBuffer(gpu_scene.seeds.handle),
                vkc::Binding::StorageBuffer(gpu_scene.textures.handle),
                vkc::Binding::StorageBuffer(gpu_scene.sobolmat.handle),
                vkc::Binding::StorageBuffer(gpu_scene.sobol_states.handle),
            ]);
            let camera_rgen: Box<dyn CameraRayGen> = {
                let any = scene.camera.as_any();
                if let Some(camera) = any.downcast_ref::<PerspectiveCamera>() {
                    Box::new(GenericCameraRayGen::<GPUPerspectiveCamera>::new(
                        ctx,
                        descriptor_pool,
                        gpu_scene,
                        &rchit_wq[0],
                        scene_set.clone(),
                        &include_spv!("spv/perspective_camera.spv"),
                        GPUPerspectiveCamera {
                            r2c: GPUTransform {
                                m4: camera.r2c.m4.into(),
                                inv_m4: camera.r2c.inv_m4.unwrap().into(),
                            },
                            c2w: GPUTransform {
                                m4: camera.c2w.m4.into(),
                                inv_m4: camera.c2w.inv_m4.unwrap().into(),
                            },
                            resolution: camera.resolution.into(),
                        },
                    ))
                } else {
                    unreachable!()
                }
            };
            let material_eval_info = MaterialEvalInfo::new(
                ctx,
                descriptor_pool,
                (scanlines_per_pass * pixels_per_scanline) as usize,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            let path_states = SOAPathState::new(
                ctx,
                descriptor_pool,
                (scanlines_per_pass * pixels_per_scanline) as usize,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            let film = TBuffer::<[f32; 4]>::new(
                ctx,
                (scene.camera.resolution().x * scene.camera.resolution().y) as usize,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            {
                let mapped = film.map_range_mut(.., vk::MemoryMapFlags::empty());
                mapped.slice.fill([0.0; 4]);
            }
            // let (film_layout, film_set) = create_descriptor_set_from_storage_buffers(
            //     &ctx.device,
            //     descriptor_pool,
            //     &[film.handle],
            //     vk::ShaderStageFlags::COMPUTE
            //         | vk::ShaderStageFlags::RAYGEN_NV
            //         | vk::ShaderStageFlags::CLOSEST_HIT_NV
            //         | vk::ShaderStageFlags::MISS_NV,
            //     ctx.allocation_callbacks,
            // );
            let shadow_queue_set = vkc::Set::Bindings(vec![
                vkc::Binding::StorageBuffer(shadow_ray_wq.ray.o.buffers[0].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ray.o.buffers[1].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ray.o.buffers[2].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ray.d.buffers[0].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ray.d.buffers[1].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ray.d.buffers[2].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ray.tmin.handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ray.tmax.handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ld.buffers[0].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ld.buffers[1].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.ld.buffers[2].handle),
                vkc::Binding::StorageBuffer(shadow_ray_wq.sid.handle),
            ]);
            let ray_queue_buffers = vkc::Set::Bindings(vec![
                vkc::Binding::StorageBuffer(ray_queue.ray.o.buffers[0].handle),
                vkc::Binding::StorageBuffer(ray_queue.ray.o.buffers[1].handle),
                vkc::Binding::StorageBuffer(ray_queue.ray.o.buffers[2].handle),
                vkc::Binding::StorageBuffer(ray_queue.ray.d.buffers[0].handle),
                vkc::Binding::StorageBuffer(ray_queue.ray.d.buffers[1].handle),
                vkc::Binding::StorageBuffer(ray_queue.ray.d.buffers[2].handle),
                vkc::Binding::StorageBuffer(ray_queue.ray.tmin.handle),
                vkc::Binding::StorageBuffer(ray_queue.ray.tmax.handle),
                vkc::Binding::StorageBuffer(ray_queue.sid.handle),
            ]);
            let path_state_set = vkc::Set::Bindings(vec![
                vkc::Binding::StorageBuffer(path_states.state.handle),
                vkc::Binding::StorageBuffer(path_states.bounce.handle),
                vkc::Binding::StorageBuffer(path_states.beta.buffers[0].handle),
                vkc::Binding::StorageBuffer(path_states.beta.buffers[1].handle),
                vkc::Binding::StorageBuffer(path_states.beta.buffers[2].handle),
                vkc::Binding::StorageBuffer(path_states.l.buffers[0].handle),
                vkc::Binding::StorageBuffer(path_states.l.buffers[1].handle),
                vkc::Binding::StorageBuffer(path_states.l.buffers[2].handle),
                vkc::Binding::StorageBuffer(path_states.pixel.handle),
            ]);
            let material_eval_info_set = vkc::Set::Bindings(vec![
                vkc::Binding::StorageBuffer(material_eval_info.wo.buffers[0].handle),
                vkc::Binding::StorageBuffer(material_eval_info.wo.buffers[1].handle),
                vkc::Binding::StorageBuffer(material_eval_info.wo.buffers[2].handle),
                vkc::Binding::StorageBuffer(material_eval_info.p.buffers[0].handle),
                vkc::Binding::StorageBuffer(material_eval_info.p.buffers[1].handle),
                vkc::Binding::StorageBuffer(material_eval_info.p.buffers[2].handle),
                vkc::Binding::StorageBuffer(material_eval_info.ng.buffers[0].handle),
                vkc::Binding::StorageBuffer(material_eval_info.ng.buffers[1].handle),
                vkc::Binding::StorageBuffer(material_eval_info.ng.buffers[2].handle),
                vkc::Binding::StorageBuffer(material_eval_info.ns.buffers[0].handle),
                vkc::Binding::StorageBuffer(material_eval_info.ns.buffers[1].handle),
                vkc::Binding::StorageBuffer(material_eval_info.ns.buffers[2].handle),
                vkc::Binding::StorageBuffer(material_eval_info.texcoords.buffers[0].handle),
                vkc::Binding::StorageBuffer(material_eval_info.texcoords.buffers[1].handle),
                vkc::Binding::StorageBuffer(material_eval_info.bsdf.handle),
            ]);
            let wq_counters = TBuffer::<u32>::new(
                ctx,
                16,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            let mesh_vertices_set = vkc::Set::StorageBufferArray(
                gpu_scene
                    .accel
                    .geometries
                    .iter()
                    .map(|g| g.mesh.vertex_buffer.handle)
                    .collect(),
            );
            let mesh_indices_set = vkc::Set::StorageBufferArray(
                gpu_scene
                    .accel
                    .geometries
                    .iter()
                    .map(|g| g.mesh.index_buffer.handle)
                    .collect(),
            );
            let mesh_normals_set = vkc::Set::StorageBufferArray(
                gpu_scene
                    .accel
                    .geometries
                    .iter()
                    .map(|g| {
                        if let Some(normals) = &g.mesh.normal_buffer {
                            normals.handle
                        } else {
                            vk::Buffer::null()
                        }
                    })
                    .collect(),
            );
            let mesh_texcoords_set = vkc::Set::StorageBufferArray(
                gpu_scene
                    .accel
                    .geometries
                    .iter()
                    .map(|g| {
                        if let Some(texcoords) = &g.mesh.texcoord_buffer {
                            texcoords.handle
                        } else {
                            vk::Buffer::null()
                        }
                    })
                    .collect(),
            );
            let image_textures_set = vkc::Set::SampledImageArray(
                gpu_scene
                    .image_textures
                    .iter()
                    .map(|img| (img.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL))
                    .collect(),
            );

            let layout = vkc::Layout {
                sets: vec![
                    vkc::Set::Bindings(vec![vkc::Binding::AccelerationStructure(
                        gpu_scene.accel.tas.accel,
                    )]),
                    ray_queue_buffers.clone(),
                    path_state_set.clone(),
                    vkc::Set::Bindings(vec![vkc::Binding::StorageBuffer(film.handle)]),
                    mesh_vertices_set.clone(),
                    mesh_indices_set.clone(),
                    mesh_normals_set.clone(),
                    mesh_texcoords_set.clone(),
                    image_textures_set.clone(),
                    scene_set.clone(),
                    material_eval_info_set.clone(),
                    vkc::Set::Bindings(vec![vkc::Binding::StorageBuffer(wq_counters.handle)]),
                ],
                push_constants: Some(16),
            };
            let intersect_kernel = vkc::Kernel::new_rchit(
                ctx,
                &gpu_scene.accel.rtx.ray_tracing,
                &include_spv!("spv/closest.rgen.spv"),
                &include_spv!("spv/closest.rchit.spv"),
                &include_spv!("spv/closest.miss.spv"),
                &layout,
            );
            let shadow_ray_kernel_spv = include_spv!("spv/shadow.rgen.spv");
            #[allow(unused_mut)]
            let mut shadow_ray_kernel_layout = vkc::Layout {
                sets: vec![
                    vkc::Set::Bindings(vec![vkc::Binding::AccelerationStructure(
                        gpu_scene.accel.tas.accel,
                    )]),
                    shadow_queue_set.clone(),
                    path_state_set.clone(),
                    vkc::Set::Bindings(vec![vkc::Binding::StorageBuffer(wq_counters.handle)]),
                ],
                push_constants: Some(16),
            };

            let shadow_ray_kernel = vkc::Kernel::new_rchit(
                ctx,
                &gpu_scene.accel.rtx.ray_tracing,
                &shadow_ray_kernel_spv,
                &include_spv!("spv/shadow.rchit.spv"),
                &include_spv!("spv/shadow.miss.spv"),
                &shadow_ray_kernel_layout,
            );

            let init_path_states_kernel = vkc::Kernel::new(
                ctx,
                &include_spv!("spv/init_path_states.spv"),
                vk::ShaderStageFlags::COMPUTE,
                &vkc::Layout {
                    sets: vec![
                        path_state_set.clone(),
                        vkc::Set::Bindings(vec![vkc::Binding::StorageBuffer(wq_counters.handle)]),
                    ],
                    push_constants: Some(16),
                },
            );

            let material_eval_kernel_spv = include_spv!("spv/material_eval.spv");
            #[allow(unused_mut)]
            let mut material_eval_kernel_layout = vkc::Layout {
                sets: vec![
                    ray_queue_buffers.clone(),
                    path_state_set.clone(),
                    material_eval_info_set.clone(),
                    vkc::Set::Bindings(vec![vkc::Binding::StorageBuffer(wq_counters.handle)]),
                    mesh_vertices_set.clone(),
                    mesh_indices_set.clone(),
                    mesh_normals_set.clone(),
                    mesh_texcoords_set.clone(),
                    image_textures_set.clone(),
                    scene_set.clone(),
                    vkc::Set::Bindings(vec![
                        vkc::Binding::StorageBuffer(gpu_scene.lights.handle),
                        vkc::Binding::StorageBuffer(gpu_scene.point_lights.handle),
                        vkc::Binding::StorageBuffer(gpu_scene.area_lights.handle),
                    ]),
                    vkc::Set::StorageBufferArray(
                        gpu_scene
                            .mesh_area_distribution
                            .iter()
                            .map(|dist| dist.alias_table.handle)
                            .collect(),
                    ),
                    vkc::Set::StorageBufferArray(
                        gpu_scene
                            .mesh_area_distribution
                            .iter()
                            .map(|dist| dist.pdf.handle)
                            .collect(),
                    ),
                    vkc::Set::Bindings(vec![
                        vkc::Binding::StorageBuffer(
                            gpu_scene.light_distribution.alias_table.handle,
                        ),
                        vkc::Binding::StorageBuffer(gpu_scene.light_distribution.pdf.handle),
                    ]),
                    shadow_queue_set.clone(),
                    #[cfg(feature = "gpu_nrc")]
                    {
                        nrc_set.clone()
                    },
                ],
                push_constants: Some(16),
            };
            let material_eval_kernel = vkc::Kernel::new(
                ctx,
                &material_eval_kernel_spv,
                vk::ShaderStageFlags::COMPUTE,
                &material_eval_kernel_layout,
            );
            let splat_film_kernel = vkc::Kernel::new(
                ctx,
                &include_spv!("spv/splat_film.spv"),
                vk::ShaderStageFlags::COMPUTE,
                &vkc::Layout {
                    sets: vec![
                        path_state_set.clone(),
                        vkc::Set::Bindings(vec![vkc::Binding::StorageBuffer(film.handle)]),
                    ],
                    push_constants: Some(16),
                },
            );
            let reset_queue_kernel =
                vkc::ComputeKernel::new(ctx, &include_spv!("spv/reset_queue.spv"));
            let reset_ray_queue_kernel =
                vkc::ComputeKernel::new(ctx, &include_spv!("spv/reset_ray_queue.spv"));
            Self {
                sampler,
                camera_rgen,
                ctx: ctx.clone(),
                gpu_scene,
                scene,
                command_buffer,
                scanlines_per_pass,
                pixels_per_scanline,
                descriptor_pool,
                path_states,
                rchit_wq,
                shadow_ray_wq,
                wq_counters,
                scene_set,
                film,
                spp,
                max_depth,
                intersect_kernel,
                reset_queue_kernel,
                reset_ray_queue_kernel,
                init_path_states_kernel,
                shadow_ray_kernel,
                material_eval_info,
                material_eval_kernel,
                splat_film_kernel,
                training_iter,
            }
        }
    }
    fn trace_shadows(&self, invoke: &InvocationContext) {
        let sbt = self.shadow_ray_kernel.sbt.as_ref().unwrap();
        let handle_size = self.gpu_scene.accel.properties.shader_group_handle_size as u32;
        let alignment = self.gpu_scene.accel.properties.shader_group_base_alignment;
        let handle_size = ((handle_size + alignment - 1) & !(alignment - 1)) as u64;
        // |[ raygen shader ]|[ hit shader  ]|[ miss shader ]|
        // |                 |               |               |
        // | 0               | 1             | 2             | 3

        let sbt_raygen_buffer = sbt.handle;
        let sbt_raygen_offset = 0;

        let sbt_miss_buffer = sbt.handle;
        let sbt_miss_offset = 2 * handle_size;
        let sbt_miss_stride = handle_size;

        let sbt_hit_buffer = sbt.handle;
        let sbt_hit_offset = 1 * handle_size;
        let sbt_hit_stride = handle_size;

        let resolution = self.scene.camera.resolution();
        self.shadow_ray_kernel.cmd_trace_rays(
            invoke.command_encoder,
            vkc::SbtRecord {
                buffer: sbt_raygen_buffer,
                offset: sbt_raygen_offset,
                stride: 0,
            },
            vkc::SbtRecord {
                buffer: sbt_miss_buffer,
                offset: sbt_miss_offset,
                stride: sbt_miss_stride,
            },
            vkc::SbtRecord {
                buffer: sbt_hit_buffer,
                offset: sbt_hit_offset,
                stride: sbt_hit_stride,
            },
            vkc::SbtRecord {
                buffer: vk::Buffer::null(),
                offset: 0,
                stride: 0,
            },
            resolution.x * invoke.scanlines_per_pass,
            1,
            1,
            Some([
                invoke.cur_scanline,
                invoke.scanlines_per_pass,
                invoke.pixels_per_scanline,
                invoke.total_scnalines,
            ]),
        );
    }
    fn intersect(&self, invoke: &InvocationContext) {
        let sbt = self.intersect_kernel.sbt.as_ref().unwrap();
        let handle_size = self.gpu_scene.accel.properties.shader_group_handle_size as u32;
        let alignment = self.gpu_scene.accel.properties.shader_group_base_alignment;
        let handle_size = ((handle_size + alignment - 1) & !(alignment - 1)) as u64;
        // |[ raygen shader ]|[ hit shader  ]|[ miss shader ]|
        // |                 |               |               |
        // | 0               | 1             | 2             | 3

        let sbt_raygen_buffer = sbt.handle;
        let sbt_raygen_offset = 0;

        let sbt_miss_buffer = sbt.handle;
        let sbt_miss_offset = 2 * handle_size;
        let sbt_miss_stride = handle_size;

        let sbt_hit_buffer = sbt.handle;
        let sbt_hit_offset = 1 * handle_size;
        let sbt_hit_stride = handle_size;

        let resolution = self.scene.camera.resolution();
        self.intersect_kernel.cmd_trace_rays(
            invoke.command_encoder,
            vkc::SbtRecord {
                buffer: sbt_raygen_buffer,
                offset: sbt_raygen_offset,
                stride: 0,
            },
            vkc::SbtRecord {
                buffer: sbt_miss_buffer,
                offset: sbt_miss_offset,
                stride: sbt_miss_stride,
            },
            vkc::SbtRecord {
                buffer: sbt_hit_buffer,
                offset: sbt_hit_offset,
                stride: sbt_hit_stride,
            },
            vkc::SbtRecord {
                buffer: vk::Buffer::null(),
                offset: 0,
                stride: 0,
            },
            resolution.x * invoke.scanlines_per_pass,
            1,
            1,
            Some([
                invoke.cur_scanline,
                invoke.scanlines_per_pass,
                invoke.pixels_per_scanline,
                invoke.total_scnalines,
            ]),
        );
    }
    fn eval_material(&self, invoke: &InvocationContext, sample_count: u32) {
        let global_size = invoke.pixels_per_scanline * invoke.scanlines_per_pass;

        self.material_eval_kernel.cmd_dispatch(
            invoke.command_encoder,
            (global_size + 255) / 256,
            1,
            1,
            Some([
                invoke.cur_scanline,
                invoke.scanlines_per_pass,
                invoke.pixels_per_scanline,
                invoke.total_scnalines,
            ]),
        )
    }
    fn splat_film(&self, invoke: &InvocationContext) {
        let global_size = invoke.pixels_per_scanline * invoke.scanlines_per_pass;
        self.splat_film_kernel.cmd_dispatch(
            invoke.command_encoder,
            (global_size + 255) / 256,
            1,
            1,
            Some([
                invoke.cur_scanline,
                invoke.scanlines_per_pass,
                invoke.pixels_per_scanline,
                invoke.total_scnalines,
            ]),
        )
    }

    fn init_path_states(&self, invoke: &InvocationContext) {
        let global_size = invoke.pixels_per_scanline * invoke.scanlines_per_pass;
        self.init_path_states_kernel.cmd_dispatch(
            invoke.command_encoder,
            (global_size + 255) / 256,
            1,
            1,
            Some([
                invoke.cur_scanline,
                invoke.scanlines_per_pass,
                invoke.pixels_per_scanline,
                invoke.total_scnalines,
            ]),
        )
    }
    fn reset_queue(&self, invoke: &InvocationContext) {
        self.reset_queue_kernel.cmd_dispatch(
            invoke.command_encoder,
            1,
            1,
            1,
            &vkc::KernelArgs {
                sets: vec![vkc::Set::Bindings(vec![vkc::Binding::StorageBuffer(
                    self.wq_counters.handle,
                )])],
                push_constants: Some([
                    invoke.cur_scanline,
                    invoke.scanlines_per_pass,
                    invoke.pixels_per_scanline,
                    invoke.total_scnalines,
                ]),
            },
        )
    }
    fn reset_ray_queue(&self, invoke: &InvocationContext) {
        self.reset_ray_queue_kernel.cmd_dispatch(
            invoke.command_encoder,
            1,
            1,
            1,
            &vkc::KernelArgs {
                sets: vec![vkc::Set::Bindings(vec![vkc::Binding::StorageBuffer(
                    self.wq_counters.handle,
                )])],
                push_constants: Some([
                    invoke.cur_scanline,
                    invoke.scanlines_per_pass,
                    invoke.pixels_per_scanline,
                    invoke.total_scnalines,
                ]),
            },
        )
    }
    fn render(&self) {
        self.render_wavefront()
    }

    fn render_wavefront(&self) {
        use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
        let resolution = self.scene.camera.resolution();
        // let mut submitted = false;
        // let mut cmd_since_last_submit = 0;
        // let mut command_encoder: *mut vkc::CommandEncoder = std::ptr::null_mut();
        let progress = ProgressBar::new(self.spp as u64);
        progress.set_draw_target(ProgressDrawTarget::stdout_with_hz(2));
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise} - {eta_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7}spp {msg}")
                .progress_chars("=>-"),
        );
        let mut profiler = Profiler::new(&self.ctx, 4096);
        for _sample_count in 0..self.spp {
            // println!("record!");
            unsafe {
                let fence = {
                    self.ctx
                        .device
                        .reset_command_buffer(
                            self.command_buffer,
                            vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                        )
                        .unwrap();
                    // cmd_since_last_submit = 0;
                    let begin_info = vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build();
                    let command_encoder = vkc::CommandEncoder::new(
                        &self.ctx.device,
                        self.command_buffer,
                        self.ctx.queue,
                        &begin_info,
                    );

                    let memory_barrier = vk::MemoryBarrier::builder()
                        .src_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        )
                        .build();
                    let mut y = 0;
                    let barrier = || {
                        self.ctx.device.cmd_pipeline_barrier(
                            self.command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::RAY_TRACING_SHADER_NV,
                            vk::DependencyFlags::empty(),
                            &[memory_barrier],
                            &[],
                            &[],
                        );
                    };
                    while y < resolution.y {
                        let command_encoder = &command_encoder;
                        let invoke = InvocationContext {
                            scanlines_per_pass: self.scanlines_per_pass,
                            cur_scanline: y,
                            command_encoder,
                            pixels_per_scanline: resolution.x,
                            total_scnalines: resolution.y,
                        };
                        profiler.profile(command_encoder, "Reset Queue", || {
                            self.reset_queue(&invoke);
                            barrier();
                        });
                        profiler.profile(command_encoder, "Init Path States", || {
                            self.init_path_states(&invoke);
                            barrier();
                        });
                        profiler.profile(command_encoder, "Camera RayGen", || {
                            self.camera_rgen.gen(&invoke);
                            barrier();
                        });
                        for _bounces in 0..self.max_depth {
                            profiler.profile(command_encoder, "Intersect Closest", || {
                                self.intersect(&invoke);
                                barrier();
                            });
                            profiler.profile(command_encoder, "Reset Queue", || {
                                self.reset_ray_queue(&invoke);
                                barrier();
                            });
                            profiler.profile(command_encoder, "Eval Material", || {
                                self.eval_material(&invoke, _sample_count);
                                barrier();
                            });
                            profiler.profile(command_encoder, "Trace Shadow Rays", || {
                                self.trace_shadows(&invoke);
                                barrier();
                            });
                        }

                        profiler.profile(command_encoder, "Splat Film", || {
                            self.splat_film(&invoke);
                            barrier();
                        });
                        y += self.scanlines_per_pass;
                    }
                    command_encoder.get_fence()
                };
                fence.wait();
                profiler.poll();
            }
            progress.inc(1);
        }

        progress.finish();
        profiler.poll();
        println!("============== KERNEL PROFILE ==============");
        println!(
            "{:20} | {:15} | {:15} | {:13} | {:13}",
            "Kernel", "Total Launches", "Total Time (ms)", "Avg Time (ms)", "Max Time (ms)"
        );
        let mut stats = profiler.all_stats();
        stats.sort_by(|a, b| {
            b.1.total_time_msec
                .partial_cmp(&a.1.total_time_msec)
                .unwrap()
        });
        for (name, stats) in stats {
            println!(
                "{:20} | {:15} | {:>15.3} | {:>13.3} | {:>13.3}",
                name,
                stats.n_launches,
                stats.total_time_msec,
                stats.avg_time_msec,
                stats.max_time_msec
            );
        }
    }
}
