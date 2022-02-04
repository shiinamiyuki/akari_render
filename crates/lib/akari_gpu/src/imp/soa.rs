use ash::vk;

use vkc::{resource::TBuffer, Context};

pub struct SOAVec<T: bytemuck::Pod, const N: usize> {
    pub buffers: [TBuffer<T>; N],
}
impl<T: bytemuck::Pod> SOAVec<T, 2> {
    pub fn new(
        ctx: &Context,
        n: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        Self {
            buffers: [
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
            ],
        }
    }
}
impl<T: bytemuck::Pod> SOAVec<T, 3> {
    pub fn new(
        ctx: &Context,
        n: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        Self {
            buffers: [
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
            ],
        }
    }
}
impl<T: bytemuck::Pod> SOAVec<T, 4> {
    pub fn new(
        ctx: &Context,
        n: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        Self {
            buffers: [
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
                TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
            ],
        }
    }
}
// pub type SOAVec<T: bytemuck::Pod, const N: usize> = [TBuffer<T>; N];
pub struct SOARay {
    pub o: SOAVec<f32, 3>,
    pub d: SOAVec<f32, 3>,
    pub tmin: TBuffer<f32>,
    pub tmax: TBuffer<f32>,
}
impl SOARay {
    pub fn new(
        ctx: &Context,
        n: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        Self {
            o: SOAVec::<f32, 3>::new(ctx, n, usage, sharing_mode, memory_property_flags),
            d: SOAVec::<f32, 3>::new(ctx, n, usage, sharing_mode, memory_property_flags),
            tmin: TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
            tmax: TBuffer::new(ctx, n, usage, sharing_mode, memory_property_flags),
        }
    }
}
pub struct SOAIntersection {
    pub prim_id: TBuffer<i32>,
    pub geom_id: TBuffer<i32>,
    pub uv: SOAVec<f32, 2>,
    pub texcoords: SOAVec<f32, 2>,
    pub ng: SOAVec<f32, 3>,
    pub ns: SOAVec<f32, 3>,
}

pub struct SOAPathState {
    pub state: TBuffer<i32>,
    pub bounce: TBuffer<i32>,
    pub beta: SOAVec<f32, 3>,
    pub l: SOAVec<f32, 3>,
    pub pixel: TBuffer<u32>,
}
impl SOAPathState {
    pub fn new(
        ctx: &Context,
        _descriptor_pool: vk::DescriptorPool,
        n: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        let state = TBuffer::<i32>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let bounce = TBuffer::<i32>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let beta = SOAVec::<f32, 3>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let l = SOAVec::<f32, 3>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let pixel = TBuffer::<u32>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        Self {
            state,
            beta,
            l,
            bounce,
            pixel,
        }
    }
}

pub struct MaterialEvalInfo {
    pub wo: SOAVec<f32, 3>,
    pub p: SOAVec<f32, 3>,
    pub ng: SOAVec<f32, 3>,
    pub ns: SOAVec<f32, 3>,
    pub texcoords: SOAVec<f32, 2>,
    pub bsdf: TBuffer<i32>,
}
impl MaterialEvalInfo {
    pub fn new(
        ctx: &Context,
        _descriptor_pool: vk::DescriptorPool,
        n: usize,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        let bsdf = TBuffer::<i32>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let wo = SOAVec::<f32, 3>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let p = SOAVec::<f32, 3>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let ng = SOAVec::<f32, 3>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let ns = SOAVec::<f32, 3>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        let texcoords = SOAVec::<f32, 2>::new(ctx, n, usage, sharing_mode, memory_property_flags);
        Self {
            bsdf,
            p,
            ng,
            ns,
            wo,
            texcoords,
        }
    }
}
