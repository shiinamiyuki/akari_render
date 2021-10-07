use crate::shape::TriangleMesh;

use ash::vk;
use vkc::{resource::TBuffer, Context};
pub struct GPUMesh {
    pub vertex_buffer: TBuffer<[f32; 3]>,
    pub index_buffer: TBuffer<[u32; 3]>,
    pub normal_buffer: Option<TBuffer<[f32; 3]>>,
    pub texcoord_buffer: Option<TBuffer<[f32; 2]>>,
    pub num_indices: u32,
    pub num_vertices: u32,
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GPUMeshInstance {
    pub geom_id: i32,
    pub bsdf_id: i32,
    pub flags: u32,
}
impl GPUMesh {
    pub fn from_triangle_mesh(ctx: &Context, mesh: &TriangleMesh) -> Self {
        let vertex_buffer = TBuffer::<[f32; 3]>::new(
            ctx,
            mesh.vertices.len(),
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        {
            let buf = vertex_buffer.map_range_mut(.., vk::MemoryMapFlags::empty());
            for i in 0..mesh.vertices.len() {
                buf.slice[i][0] = mesh.vertices[i][0];
                buf.slice[i][1] = mesh.vertices[i][1];
                buf.slice[i][2] = mesh.vertices[i][2];
            }
        }
        let normal_buffer = if mesh.normals.is_empty() {
            None
        } else {
            let normal_buffer = TBuffer::<[f32; 3]>::new(
                ctx,
                mesh.normals.len(),
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            {
                let buf = normal_buffer.map_range_mut(.., vk::MemoryMapFlags::empty());
                for i in 0..mesh.normals.len() {
                    buf.slice[i][0] = mesh.normals[i][0];
                    buf.slice[i][1] = mesh.normals[i][1];
                    buf.slice[i][2] = mesh.normals[i][2];
                }
            }
            Some(normal_buffer)
        };
        let texcoord_buffer = if mesh.texcoords.is_empty() {
            None
        } else {
            let texcoord_buffer = TBuffer::<[f32; 2]>::new(
                ctx,
                mesh.texcoords.len(),
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            {
                let buf = texcoord_buffer.map_range_mut(.., vk::MemoryMapFlags::empty());
                for i in 0..mesh.texcoords.len() {
                    buf.slice[i][0] = mesh.texcoords[i][0];
                    buf.slice[i][1] = mesh.texcoords[i][1];
                }
            }
            Some(texcoord_buffer)
        };
        let index_buffer = TBuffer::<[u32; 3]>::new(
            ctx,
            mesh.indices.len(),
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        {
            let buf = index_buffer.map_range_mut(.., vk::MemoryMapFlags::empty());
            for i in 0..mesh.indices.len() {
                buf.slice[i][0] = mesh.indices[i][0];
                buf.slice[i][1] = mesh.indices[i][1];
                buf.slice[i][2] = mesh.indices[i][2];
            }
        }
        Self {
            vertex_buffer,
            index_buffer,
            texcoord_buffer,
            normal_buffer,
            num_indices: mesh.indices.len() as u32,
            num_vertices: mesh.vertices.len() as u32,
        }
    }
}
