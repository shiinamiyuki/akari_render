use std::rc::Rc;

use super::mesh::GPUMesh;
use super::mesh::GPUMeshInstance;
use crate::scene::Scene;
use crate::shape::AggregateProxy;
use crate::shape::MeshInstanceProxy;
use crate::shape::TriangleMesh;
use crate::*;
use ash::extensions::nv;
use ash::vk;
use bytemuck::Pod;
use bytemuck::Zeroable;
use vkc::resource::find_memorytype_index;
use vkc::resource::TBuffer;
use vkc::Context;
// maybe as fallback
// #[derive(Clone, Copy, Pod, Zeroable)]
// #[repr(C)]
// pub struct GPUAABB {
//     pub pmin: [f32; 4],
//     pub pmax: [f32; 4],
// }

// #[derive(Clone, Copy, Pod, Zeroable)]
// #[repr(C)]
// pub struct GPUBVHNode {
//     pub aabb: GPUAABB,
//     pub axis: u32,
//     pub first: u32,
//     pub count: u32,
//     pub left: u32,
//     pub right: u32,
// }

// pub struct GPUBVH {
//     pub nodes: TBuffer<GPUBVHNode>,
//     pub references: TBuffer<u32>,
// }

// pub struct GPUTopLevelBVH {
//     pub top: GPUBVH,
//     pub bvhes: Vec<GPUBVH>,
// }

pub struct RTX {
    ctx: Context,
    pub ray_tracing: Rc<nv::RayTracing>,
}
pub struct GPUAccel {
    ctx: Context,
    pub properties: vk::PhysicalDeviceRayTracingPropertiesNV,
    pub rtx: RTX,
    pub tas: TAS,
    pub geometries: Vec<GPUGeometry>,
    pub descriptor_pool: vk::DescriptorPool,
    pub instances: Vec<GPUMeshInstance>,
    pub shape_to_instance: HashMap<u64, u32>,
}
pub struct GPUGeometry {
    #[allow(dead_code)]
    ctx: Context,
    pub geometry: vk::GeometryNV,
    pub gas: vk::AccelerationStructureNV,
    pub mesh: GPUMesh,
    pub memory: vk::DeviceMemory,
}
pub struct TAS {
    pub accel: vk::AccelerationStructureNV,
    memory: vk::DeviceMemory,
    instances: Vec<GeometryInstance>,
    instance_buffer: TBuffer<GeometryInstance>,
}

#[repr(C)]
#[derive(Clone, Debug, Copy, Pod, Zeroable)]
pub struct GeometryInstance {
    transform: [f32; 12],
    instance_id_and_mask: u32,
    instance_offset_and_flags: u32,
    acceleration_handle: u64,
}

impl GeometryInstance {
    fn new(
        transform: [f32; 12],
        id: u32,
        mask: u8,
        offset: u32,
        flags: vk::GeometryInstanceFlagsNV,
        acceleration_handle: u64,
    ) -> Self {
        let mut instance = GeometryInstance {
            transform,
            instance_id_and_mask: 0,
            instance_offset_and_flags: 0,
            acceleration_handle,
        };
        instance.set_id(id);
        instance.set_mask(mask);
        instance.set_offset(offset);
        instance.set_flags(flags);
        instance
    }

    fn set_id(&mut self, id: u32) {
        let id = id & 0x00ffffff;
        self.instance_id_and_mask |= id;
    }

    fn set_mask(&mut self, mask: u8) {
        let mask = mask as u32;
        self.instance_id_and_mask |= mask << 24;
    }

    fn set_offset(&mut self, offset: u32) {
        let offset = offset & 0x00ffffff;
        self.instance_offset_and_flags |= offset;
    }

    fn set_flags(&mut self, flags: vk::GeometryInstanceFlagsNV) {
        let flags = flags.as_raw() as u32;
        self.instance_offset_and_flags |= flags << 24;
    }
}

pub fn mat4_to_rtx_transform(transform: &glm::Mat4) -> [f32; 12] {
    [
        transform[(0, 0)],
        transform[(0, 1)],
        transform[(0, 2)],
        transform[(0, 3)],
        transform[(1, 0)],
        transform[(1, 1)],
        transform[(1, 2)],
        transform[(1, 3)],
        transform[(2, 0)],
        transform[(2, 1)],
        transform[(2, 2)],
        transform[(2, 3)],
    ]
}
impl RTX {
    fn create_instance(
        &self,
        geometry: &GPUGeometry,
        transform: &glm::Mat4,
        id: u32,
    ) -> GeometryInstance {
        assert_eq!(
            std::mem::size_of::<GeometryInstance>(),
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()
        );
        unsafe {
            GeometryInstance::new(
                mat4_to_rtx_transform(transform),
                id,
                0xff,
                0,
                vk::GeometryInstanceFlagsNV::TRIANGLE_CULL_DISABLE_NV,
                self.ray_tracing
                    .get_acceleration_structure_handle(geometry.gas)
                    .unwrap(),
            )
        }
    }
    fn create_gas_from_mesh(&self, mesh: &TriangleMesh) -> GPUGeometry {
        unsafe {
            let ctx = &self.ctx;
            let mesh = GPUMesh::from_triangle_mesh(ctx, mesh);

            let geometry = vk::GeometryNV::builder()
                .geometry_type(vk::GeometryTypeNV::TRIANGLES)
                .geometry(
                    *vk::GeometryDataNV::builder().triangles(
                        vk::GeometryTrianglesNV::builder()
                            .vertex_data(mesh.vertex_buffer.handle)
                            .vertex_offset(0)
                            .vertex_count(mesh.num_vertices)
                            .vertex_stride(std::mem::size_of::<[f32; 3]>() as u64)
                            .vertex_format(vk::Format::R32G32B32_SFLOAT)
                            .index_data(mesh.index_buffer.handle)
                            .index_offset(0)
                            .index_count(mesh.num_indices * 3)
                            .index_type(vk::IndexType::UINT32)
                            .build(),
                    ),
                )
                .flags(vk::GeometryFlagsNV::OPAQUE)
                .build();

            let accel_info = vk::AccelerationStructureCreateInfoNV::builder()
                .compacted_size(0)
                .info(
                    vk::AccelerationStructureInfoNV::builder()
                        .ty(vk::AccelerationStructureTypeNV::BOTTOM_LEVEL)
                        .geometries(&[geometry])
                        .flags(vk::BuildAccelerationStructureFlagsNV::PREFER_FAST_TRACE)
                        .build(),
                )
                .build();

            let gas = self
                .ray_tracing
                .create_acceleration_structure(&accel_info, ctx.allocation_callbacks.as_ref())
                .unwrap();

            let memory_requirements = self
                .ray_tracing
                .get_acceleration_structure_memory_requirements(
                    &vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                        .acceleration_structure(gas)
                        .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::OBJECT)
                        .build(),
                );

            let memory = ctx
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(memory_requirements.memory_requirements.size)
                        .memory_type_index(
                            find_memorytype_index(
                                &memory_requirements.memory_requirements,
                                &ctx.device_memory_properties,
                                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            )
                            .unwrap(),
                        )
                        .build(),
                    ctx.allocation_callbacks.as_ref(),
                )
                .unwrap();

            self.ray_tracing
                .bind_acceleration_structure_memory(&[
                    vk::BindAccelerationStructureMemoryInfoNV::builder()
                        .acceleration_structure(gas)
                        .memory(memory)
                        .build(),
                ])
                .unwrap();

            GPUGeometry {
                mesh,
                memory,
                gas,
                geometry,
                ctx: ctx.clone(),
            }
        }
    }
    pub fn create_instance_buffer(
        &self,
        instances: &Vec<GeometryInstance>,
    ) -> TBuffer<GeometryInstance> {
        let buffer: TBuffer<GeometryInstance> = TBuffer::new(
            &self.ctx,
            instances.len(),
            vk::BufferUsageFlags::RAY_TRACING_NV
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        buffer.store(instances);
        buffer
    }
    fn compute_gas_scratch_buffer_size(&self, geometries: &[GPUGeometry]) -> u64 {
        unsafe {
            geometries
                .iter()
                .map(|geometry| {
                    let requirements = self
                        .ray_tracing
                        .get_acceleration_structure_memory_requirements(
                        &vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                            .acceleration_structure(geometry.gas)
                            .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::BUILD_SCRATCH)
                            .build(),
                    );
                    requirements.memory_requirements.size
                })
                .max()
                .unwrap()
        }
    }
    fn compute_tas_scratch_buffer_size(&self, tas: &TAS) -> u64 {
        unsafe {
            let requirements = self
                .ray_tracing
                .get_acceleration_structure_memory_requirements(
                    &vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                        .acceleration_structure(tas.accel)
                        .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::BUILD_SCRATCH)
                        .build(),
                );
            requirements.memory_requirements.size
        }
    }
    // assume command_buffer has begun
    fn cmd_build_gas(
        &self,
        scratch_buffer: &TBuffer<u8>,
        command_buffer: vk::CommandBuffer,
        geometry: &GPUGeometry,
        memory_barrier: vk::MemoryBarrier,
    ) {
        unsafe {
            self.ray_tracing.cmd_build_acceleration_structure(
                command_buffer,
                &vk::AccelerationStructureInfoNV::builder()
                    .ty(vk::AccelerationStructureTypeNV::BOTTOM_LEVEL)
                    .geometries(&[geometry.geometry])
                    .flags(vk::BuildAccelerationStructureFlagsNV::PREFER_FAST_TRACE)
                    .build(),
                vk::Buffer::null(),
                0,
                false,
                geometry.gas,
                vk::AccelerationStructureNV::null(),
                scratch_buffer.handle,
                0,
            );
            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }
    fn cmd_build_tas(
        &self,
        scratch_buffer: &TBuffer<u8>,
        command_buffer: vk::CommandBuffer,
        tas: &TAS,
        memory_barrier: vk::MemoryBarrier,
    ) {
        unsafe {
            self.ray_tracing.cmd_build_acceleration_structure(
                command_buffer,
                &vk::AccelerationStructureInfoNV::builder()
                    .ty(vk::AccelerationStructureTypeNV::TOP_LEVEL)
                    .instance_count(tas.instances.len() as u32)
                    .flags(vk::BuildAccelerationStructureFlagsNV::PREFER_FAST_TRACE)
                    .build(),
                tas.instance_buffer.handle,
                0,
                false,
                tas.accel,
                vk::AccelerationStructureNV::null(),
                scratch_buffer.handle,
                0,
            );
            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_NV,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }
}
impl GPUAccel {
    pub fn new(ctx: &Context, scene: &Scene) -> Self {
        unsafe {
            let props_rt = nv::RayTracing::get_properties(&ctx.instance, ctx.pdevice);
            println!("NV Ray Tracing Properties:");
            println!(
                " shader_group_handle_size: {}",
                props_rt.shader_group_handle_size
            );
            println!(" max_recursion_depth: {}", props_rt.max_recursion_depth);
            println!(
                " max_shader_group_stride: {}",
                props_rt.max_shader_group_stride
            );
            println!(
                " shader_group_base_alignment: {}",
                props_rt.shader_group_base_alignment
            );
            println!(" max_geometry_count: {}", props_rt.max_geometry_count);
            println!(" max_instance_count: {}", props_rt.max_instance_count);
            println!(" max_triangle_count: {}", props_rt.max_triangle_count);
            println!(
                " max_descriptor_set_acceleration_structures: {}",
                props_rt.max_descriptor_set_acceleration_structures
            );
            let rtx = RTX {
                ray_tracing: Rc::new(nv::RayTracing::new(&ctx.instance, &ctx.device)),
                ctx: ctx.clone(),
            };
            let aggregate: Option<&AggregateProxy> = downcast_ref(scene.shape.as_ref());
            let aggragate = aggregate.unwrap();
            let mut shape_to_geometry = HashMap::new();
            let geometries: Vec<_> = scene
                .meshes
                .iter()
                .enumerate()
                .map(|(i, mesh)| -> GPUGeometry {
                    let geometry = rtx.create_gas_from_mesh(mesh);
                    let addr = Arc::into_raw(mesh.clone()).cast::<()>() as u64;
                    shape_to_geometry.insert(addr, i);
                    geometry
                })
                .collect();
            log::info!("building GAS");
            let mut mesh_instances: Vec<GPUMeshInstance> = vec![];
            let mut shape_to_instance = HashMap::new();
            let instances: Vec<GeometryInstance> = aggragate
                .shapes
                .iter()
                .enumerate()
                .map(|(instane_id, shape)| {
                    let proxy: Option<&MeshInstanceProxy> = downcast_ref(shape.as_ref());
                    if let Some(proxy) = proxy {
                        let addr = Arc::into_raw(proxy.mesh.clone()).cast::<()>() as u64;
                        let geometry_id = shape_to_geometry.get(&addr).unwrap();
                        let geometry = &geometries[*geometry_id];
                        let mut flags = 0u32;
                        if geometry.mesh.normal_buffer.is_some() {
                            flags |= 1;
                        }
                        if geometry.mesh.texcoord_buffer.is_some() {
                            flags |= 2;
                        }
                        mesh_instances.push(GPUMeshInstance {
                            geom_id: *geometry_id as i32,
                            bsdf_id: -1,
                            flags,
                        });
                        shape_to_instance.insert(
                            Arc::into_raw(shape.clone()).cast::<()>() as u64,
                            instane_id as u32,
                        );
                        rtx.create_instance(geometry, &glm::identity(), instane_id as u32)
                    } else {
                        panic!("only triangle mesh is supported on gpu");
                    }
                })
                .collect();

            let instance_buffer = rtx.create_instance_buffer(&instances);
            let accel_info = vk::AccelerationStructureCreateInfoNV::builder()
                .compacted_size(0)
                .info(
                    vk::AccelerationStructureInfoNV::builder()
                        .ty(vk::AccelerationStructureTypeNV::TOP_LEVEL)
                        .instance_count(instances.len() as u32)
                        .flags(vk::BuildAccelerationStructureFlagsNV::PREFER_FAST_TRACE)
                        .build(),
                )
                .build();
            log::info!("building TAS");
            let tas = rtx
                .ray_tracing
                .create_acceleration_structure(&accel_info, None)
                .unwrap();
            let memory_requirements = rtx
                .ray_tracing
                .get_acceleration_structure_memory_requirements(
                    &vk::AccelerationStructureMemoryRequirementsInfoNV::builder()
                        .acceleration_structure(tas)
                        .ty(vk::AccelerationStructureMemoryRequirementsTypeNV::OBJECT)
                        .build(),
                );
            let tas_memory = ctx
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(memory_requirements.memory_requirements.size)
                        .memory_type_index(
                            find_memorytype_index(
                                &memory_requirements.memory_requirements,
                                &ctx.device_memory_properties,
                                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            )
                            .unwrap(),
                        )
                        .build(),
                    None,
                )
                .unwrap();
            rtx.ray_tracing
                .bind_acceleration_structure_memory(&[
                    vk::BindAccelerationStructureMemoryInfoNV::builder()
                        .acceleration_structure(tas)
                        .memory(tas_memory)
                        .build(),
                ])
                .unwrap();
            let tas = TAS {
                accel: tas,
                memory: tas_memory,
                instance_buffer,
                instances,
            };

            let scratch_buffer_size = {
                rtx.compute_gas_scratch_buffer_size(geometries.as_slice())
                    .max(rtx.compute_tas_scratch_buffer_size(&tas))
            };
            println!("scratch_buffer_size={}", scratch_buffer_size);
            let scratch_buffer = TBuffer::<u8>::new(
                ctx,
                scratch_buffer_size as usize,
                vk::BufferUsageFlags::RAY_TRACING_NV,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            log::info!("submitting commands");
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(ctx.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .build();

            let command_buffers = ctx.device.allocate_command_buffers(&allocate_info).unwrap();
            let memory_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV,
                )
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_NV
                        | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_NV,
                )
                .build();
            let build_command_buffer = command_buffers[0];
            ctx.device
                .begin_command_buffer(
                    build_command_buffer,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                )
                .unwrap();

            for geometry in &geometries {
                // log::info!("cmd_build_gas");
                rtx.cmd_build_gas(
                    &scratch_buffer,
                    build_command_buffer,
                    geometry,
                    memory_barrier,
                );
            }
            // log::info!("cmd_build_tas");
            rtx.cmd_build_tas(&scratch_buffer, build_command_buffer, &tas, memory_barrier);
            ctx.device.end_command_buffer(build_command_buffer).unwrap();
            ctx.device
                .queue_submit(
                    ctx.queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[build_command_buffer])
                        .build()],
                    vk::Fence::null(),
                )
                .expect("queue submit failed.");
            match ctx.device.queue_wait_idle(ctx.queue) {
                Ok(_) => println!("Successfully built acceleration structures"),
                Err(err) => {
                    println!("Failed to build acceleration structures: {:?}", err);
                    panic!("GPU ERROR");
                }
            }
            let descriptor_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
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
            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&descriptor_sizes)
                .max_sets(64);
            let descriptor_pool = ctx
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap();
            ctx.device
                .free_command_buffers(ctx.pool, &[build_command_buffer]);
            Self {
                ctx: ctx.clone(),
                rtx,
                geometries,
                tas,
                descriptor_pool,
                instances: mesh_instances,
                properties: props_rt,
                shape_to_instance,
            }
        }
    }
}
impl Drop for GPUAccel {
    fn drop(&mut self) {
        unsafe {
            self.ctx.device.destroy_descriptor_pool(
                self.descriptor_pool,
                self.ctx.allocation_callbacks.as_ref(),
            );
            self.ctx
                .device
                .free_memory(self.tas.memory, self.ctx.allocation_callbacks.as_ref());
            self.rtx.ray_tracing.destroy_acceleration_structure(
                self.tas.accel,
                self.ctx.allocation_callbacks.as_ref(),
            );
            for geometry in &self.geometries {
                self.ctx
                    .device
                    .free_memory(geometry.memory, self.ctx.allocation_callbacks.as_ref());
                self.rtx.ray_tracing.destroy_acceleration_structure(
                    geometry.gas,
                    self.ctx.allocation_callbacks.as_ref(),
                );
            }
        }
    }
}
