use std::collections::{HashMap, VecDeque};
use std::mem::size_of_val;
use std::sync::Arc;

use crate::bsdf::{Bsdf, GPUBsdfProxy};
use crate::light::{AreaLight, PointLight, PowerLightDistribution};
use crate::scene::Scene;
use crate::shape::{GPUAggregate, GPUTriangleMeshInstanceProxy};
use crate::sobolmat::{SOBOL_BITS, SOBOL_MATRIX, SOBOL_MAX_DIMENSION};
use crate::texture::{ConstantTexture, ImageTexture, Texture};
use crate::{downcast_ref, Spectrum};

use super::accel::GPUAccel;
use super::mesh::GPUMeshInstance;
use ash::vk;
use rand::{thread_rng, Rng};
use vkc::resource::Image;
use vkc::{resource::TBuffer, Context};
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUTexture {
    pub ty: i32,
    pub image_tex_id: i32,
    pub _pad0: [i32; 2],
    pub data: [f32; 4],
}
impl GPUTexture {
    pub const TYPE_FLOAT: i32 = 0;
    pub const TYPE_SPECTRUM: i32 = 1;
    pub const TEXTURE_FLOAT_IMAGE: i32 = 2;
    pub const TEXTURE_SPECTRUM_IMAGE: i32 = 2;
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUPointLight {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub texture: GPUTexture,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUAreaLight {
    pub instance_id: u32,
    pub area_dist_id: u32,
    pub _pad: [u32; 2],
    pub emission: GPUTexture,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPULight {
    pub ty: u32,
    pub index: u32,
}
impl GPULight {
    pub const TYPE_POINT: u32 = 0;
    pub const TYPE_MESH: u32 = 1;
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct AliasTableEntry {
    pub j: u32,
    pub t: f32,
}
pub type AliasTable = Vec<AliasTableEntry>;
pub type GPUAliasTable = TBuffer<AliasTableEntry>;

pub fn create_alias_table(weights: &[f32]) -> AliasTable {
    assert!(weights.len() >= 1);
    let sum: f32 = weights.iter().map(|x| *x).sum::<f32>();
    let mut prob: Vec<_> = weights
        .iter()
        .map(|x| *x / sum * (weights.len() as f32))
        .collect();
    let mut small = VecDeque::new();
    let mut large = VecDeque::new();
    for (i, p) in prob.iter().enumerate() {
        if *p >= 1.0 {
            large.push_back(i);
        } else {
            small.push_back(i);
        }
    }
    let mut table = vec![AliasTableEntry::default(); weights.len()];
    while !small.is_empty() && !large.is_empty() {
        let l = small.pop_front().unwrap();
        let g = large.pop_front().unwrap();
        table[l].t = prob[l];
        table[l].j = g as u32;
        prob[g] = (prob[g] + prob[l]) - 1.0;
        if prob[g] < 1.0 {
            small.push_back(g);
        } else {
            large.push_back(g);
        }
    }
    while !large.is_empty() {
        let g = large.pop_front().unwrap();
        table[g].t = 1.0;
        table[g].j = g as u32;
    }
    while !small.is_empty() {
        let l = small.pop_front().unwrap();
        table[l].t = 1.0;
        table[l].j = l as u32;
    }
    table
}
// pub fn create_alias_table(weights: &[f32]) -> AliasTable {
//     assert!(weights.len() >= 1);
//     let avg: f32 = weights.iter().map(|x| *x).sum::<f32>() / (weights.len() as f32);
//     let mut f: Vec<_> = weights
//         .iter()
//         .enumerate()
//         .map(|(i, x)| (i as u32, *x))
//         .collect();
//     let mut table_tmp: Vec<(u32, u32, f32)> = vec![];
//     while f.len() > 1 {
//         f.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
//         let wi = f[0].1;
//         table_tmp.push((f[0].0, f.last().unwrap().0, wi / avg));
//         f.last_mut().unwrap().1 -= avg - wi;
//         let last = f.len() - 1;
//         f.swap(0, last);
//         f.pop();
//     }
//     table_tmp.push((f[0].0, f[0].0, 0.0));
//     let mut table = vec![AliasTableEntry::default(); table_tmp.len()];
//     for entry in table_tmp {
//         table[entry.0 as usize] = AliasTableEntry {
//             j: entry.1,
//             t: entry.2,
//         };
//     }
//     table
// }
pub struct GPUDistribution1D {
    pub alias_table: GPUAliasTable,
    pub pdf: TBuffer<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUBsdf {
    pub color: GPUTexture,
    pub metallic: GPUTexture,
    pub roughness: GPUTexture,
    pub emission: GPUTexture,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUSobolSamplerState {
    pub rotation: u32,
    pub sample: u32,
    pub dimension: u32,
}
pub struct GPUScene {
    pub accel: GPUAccel,
    pub instances: TBuffer<GPUMeshInstance>,
    pub seeds: TBuffer<u32>,
    pub sobolmat: TBuffer<u32>,
    pub sobol_states: TBuffer<GPUSobolSamplerState>,
    pub ctx: Context,
    pub image_textures: Vec<Image>,
    pub bsdfs: TBuffer<GPUBsdf>,
    pub textures: TBuffer<GPUTexture>,
    pub point_lights: TBuffer<GPUPointLight>,
    pub area_lights: TBuffer<GPUAreaLight>,
    pub lights: TBuffer<GPULight>,
    pub light_distribution: GPUDistribution1D,
    pub mesh_area_distribution: Vec<GPUDistribution1D>,
}

struct GPUSceneBuilder {
    bsdfs: Vec<GPUBsdf>,
    textures: Vec<GPUTexture>,
    image_textures: Vec<Image>,
    bsdf_to_id: HashMap<u64, i32>,
    texture_to_id: HashMap<u64, i32>,
}
impl GPUScene {
    pub fn new(ctx: &Context, scene: &Scene) -> Self {
        let mut accel = GPUAccel::new(ctx, scene);
        let npixels: u32 = scene.camera.resolution().x * scene.camera.resolution().y;
        let mut builder = GPUSceneBuilder {
            bsdf_to_id: HashMap::new(),
            texture_to_id: HashMap::new(),
            bsdfs: vec![],
            textures: vec![],
            image_textures: vec![],
        };
        let get_texture = |builder: &mut GPUSceneBuilder,
                           texture: &Arc<dyn Texture>|
         -> GPUTexture {
            let map = &mut builder.texture_to_id;
            let addr = Arc::as_ptr(texture).cast::<()>() as u64;
            let id = if let Some(id) = map.get(&addr) {
                *id
            } else {
                let any = texture.as_any();
                let gpu_texture: GPUTexture =
                    if let Some(texture) = any.downcast_ref::<ConstantTexture<f32>>() {
                        GPUTexture {
                            ty: GPUTexture::TYPE_FLOAT,
                            _pad0: [0; 2],
                            data: [texture.value; 4],
                            image_tex_id: -1,
                        }
                    } else if let Some(texture) = any.downcast_ref::<ConstantTexture<f64>>() {
                        log::warn!(
                        "ConstantTexture<f64> is implitly converted to ConstantTexture<f32> on gpu"
                    );
                        GPUTexture {
                            ty: GPUTexture::TYPE_FLOAT,
                            _pad0: [0; 2],
                            data: [texture.value as f32; 4],
                            image_tex_id: -1,
                        }
                    } else if let Some(texture) = any.downcast_ref::<ConstantTexture<Spectrum>>() {
                        assert!(Spectrum::N_SAMPLES == 3, "currently only rgb is supported");
                        GPUTexture {
                            ty: GPUTexture::TYPE_SPECTRUM,
                            _pad0: [0; 2],
                            data: [
                                texture.value.samples[0],
                                texture.value.samples[1],
                                texture.value.samples[2],
                                0.0,
                            ],
                            image_tex_id: -1,
                        }
                    } else if let Some(texture) = any.downcast_ref::<ImageTexture<f32>>() {
                        let pixels: Vec<f32> = texture
                            .as_slice()
                            .iter()
                            .flat_map(|px| -> [f32; 4] { [*px, *px, *px, 1.0] })
                            .collect();
                        let image_tex = vkc::resource::Image::from_data(
                            ctx,
                            bytemuck::cast_slice(&pixels),
                            vk::Extent2D {
                                width: texture.width(),
                                height: texture.height(),
                            },
                            vk::Format::R32G32B32A32_SFLOAT,
                        );

                        let id = builder.image_textures.len();
                        builder.image_textures.push(image_tex);
                        GPUTexture {
                            ty: GPUTexture::TEXTURE_FLOAT_IMAGE,
                            _pad0: [0; 2],
                            data: [1.0; 4],
                            image_tex_id: id as i32,
                        }
                    } else if let Some(_texture) = any.downcast_ref::<ImageTexture<f64>>() {
                        panic!("f64 texture not supported")
                    } else if let Some(texture) = any.downcast_ref::<ImageTexture<Spectrum>>() {
                        let pixels: Vec<f32> = texture
                            .as_slice()
                            .iter()
                            .flat_map(|px| -> [f32; 4] {
                                [px.samples[0], px.samples[1], px.samples[2], 1.0]
                            })
                            .collect();
                        let image_tex = vkc::resource::Image::from_data(
                            ctx,
                            bytemuck::cast_slice(&pixels),
                            vk::Extent2D {
                                width: texture.width(),
                                height: texture.height(),
                            },
                            vk::Format::R32G32B32A32_SFLOAT,
                        );
                        let id = builder.image_textures.len();
                        builder.image_textures.push(image_tex);
                        GPUTexture {
                            ty: GPUTexture::TEXTURE_SPECTRUM_IMAGE,
                            _pad0: [0; 2],
                            data: [1.0; 4],
                            image_tex_id: id as i32,
                        }
                    } else {
                        unreachable!()
                    };
                let id = builder.textures.len() as i32;
                builder.textures.push(gpu_texture);
                builder.texture_to_id.insert(addr, id);
                id
            };
            builder.textures[id as usize]
        };
        #[allow(dead_code)]
        let get_bsdf_id = |builder: &mut GPUSceneBuilder, bsdf: &Arc<dyn Bsdf>| -> i32 {
            let map = &mut builder.bsdf_to_id;
            let bsdfs = &mut builder.bsdfs;
            let addr = Arc::as_ptr(bsdf).cast::<()>() as u64;
            if let Some(id) = map.get(&addr) {
                *id
            } else {
                let proxy: Option<&GPUBsdfProxy> = downcast_ref(bsdf.as_ref());
                let proxy = proxy.unwrap();
                let id = bsdfs.len() as i32;
                map.insert(addr, id);
                // let map = ();
                // let bsdfs = ();
                drop(map);
                drop(bsdfs);
                let bsdf = GPUBsdf {
                    color: get_texture(builder, &proxy.color),
                    roughness: get_texture(builder, &proxy.roughness),
                    metallic: get_texture(builder, &proxy.metallic),
                    emission: get_texture(builder, &proxy.emission),
                };
                let bsdfs = &mut builder.bsdfs;
                bsdfs.push(bsdf);
                id
            }
        };
        {
            let aggregate: Option<&GPUAggregate> = downcast_ref(scene.shape.as_ref());
            let aggragate = aggregate.unwrap();
            let instances = &mut accel.instances;
            for (i, instance) in instances.iter_mut().enumerate() {
                let proxy: Option<&GPUTriangleMeshInstanceProxy> =
                    downcast_ref(aggragate.shapes[i].as_ref());
                let proxy = proxy.unwrap();
                let id = get_bsdf_id(&mut builder, &proxy.bsdf);
                instance.bsdf_id = id;
            }
        }
        let light_dist = {
            if let Some(dist) = scene
                .light_distr
                .as_any()
                .downcast_ref::<PowerLightDistribution>()
            {
                let alias_table = create_alias_table(dist.pdf());
                let gpu_alias_table = TBuffer::<AliasTableEntry>::new(
                    ctx,
                    alias_table.len(),
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::SharingMode::EXCLUSIVE,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                );
                {
                    let mapped = gpu_alias_table.map_range_mut(.., vk::MemoryMapFlags::empty());
                    mapped.slice.copy_from_slice(&alias_table);
                }

                let buf = TBuffer::<f32>::new(
                    ctx,
                    scene.lights.len(),
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::SharingMode::EXCLUSIVE,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                );
                {
                    let mapped = buf.map_range_mut(.., vk::MemoryMapFlags::empty());
                    mapped.slice.copy_from_slice(dist.pdf());
                }
                GPUDistribution1D {
                    pdf: buf,
                    alias_table: gpu_alias_table,
                }
            } else {
                unimplemented!()
            }
        };
        let (lights, point_lights, area_lights, area_dist) = {
            let mut lights: Vec<GPULight> = vec![];
            let mut point_lights: Vec<GPUPointLight> = vec![];
            let mut area_lights: Vec<GPUAreaLight> = vec![];
            let mut area_dist: Vec<GPUDistribution1D> = vec![];
            for light in &scene.lights {
                let light = light.as_any();
                if let Some(point) = light.downcast_ref::<PointLight>() {
                    point_lights.push(GPUPointLight {
                        pos: point.position.into(),
                        _pad0: 0.0,
                        texture: get_texture(&mut builder, &point.emission),
                    });
                    lights.push(GPULight {
                        ty: GPULight::TYPE_POINT,
                        index: point_lights.len() as u32 - 1,
                    });
                } else if let Some(area) = light.downcast_ref::<AreaLight>() {
                    let proxy = area
                        .shape
                        .as_any()
                        .downcast_ref::<GPUTriangleMeshInstanceProxy>()
                        .unwrap();
                    let dist = proxy.mesh.area_distribution();
                    let alias_table = create_alias_table(&dist.pmf);
                    let gpu_alias_table = TBuffer::<AliasTableEntry>::new(
                        ctx,
                        alias_table.len(),
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::TRANSFER_SRC,
                        vk::SharingMode::EXCLUSIVE,
                        vk::MemoryPropertyFlags::HOST_COHERENT
                            | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    );
                    {
                        let mapped = gpu_alias_table.map_range_mut(.., vk::MemoryMapFlags::empty());
                        mapped.slice.copy_from_slice(&alias_table);
                    }
                    let pdf_buf = TBuffer::<f32>::new(
                        ctx,
                        alias_table.len(),
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::TRANSFER_SRC,
                        vk::SharingMode::EXCLUSIVE,
                        vk::MemoryPropertyFlags::HOST_COHERENT
                            | vk::MemoryPropertyFlags::HOST_VISIBLE,
                    );
                    {
                        let mapped = pdf_buf.map_range_mut(.., vk::MemoryMapFlags::empty());
                        mapped.slice.copy_from_slice(&dist.pmf);
                    }
                    area_lights.push(GPUAreaLight {
                        emission: get_texture(&mut builder, &area.emission),
                        instance_id: *accel
                            .shape_to_instance
                            .get(&(Arc::into_raw(area.shape.clone()).cast::<()>() as u64))
                            .unwrap(),
                        area_dist_id: area_dist.len() as u32,
                        _pad: [0; 2],
                    });
                    lights.push(GPULight {
                        ty: GPULight::TYPE_MESH,
                        index: area_lights.len() as u32 - 1,
                    });
                    area_dist.push(GPUDistribution1D {
                        alias_table: gpu_alias_table,
                        pdf: pdf_buf,
                    });
                }
            }
            let point_lights_buf = TBuffer::<GPUPointLight>::new(
                ctx,
                point_lights.len(),
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            if !point_lights_buf.is_empty() {
                let mapped = point_lights_buf.map_range_mut(.., vk::MemoryMapFlags::empty());
                mapped.slice.copy_from_slice(point_lights.as_slice());
            }
            let area_lights_buf = TBuffer::<GPUAreaLight>::new(
                ctx,
                area_lights.len(),
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            if !area_lights_buf.is_empty() {
                let mapped = area_lights_buf.map_range_mut(.., vk::MemoryMapFlags::empty());
                mapped.slice.copy_from_slice(area_lights.as_slice());
            }
            let lights_buf = TBuffer::<GPULight>::new(
                ctx,
                lights.len(),
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            {
                let mapped = lights_buf.map_range_mut(.., vk::MemoryMapFlags::empty());
                mapped.slice.copy_from_slice(lights.as_slice());
            }
            log::info!(
                "area lights: {}, point lights: {}",
                area_lights.len(),
                point_lights.len()
            );
            (lights_buf, point_lights_buf, area_lights_buf, area_dist)
        };

        let seeds = {
            let seeds: TBuffer<u32> = TBuffer::new(
                ctx,
                npixels as usize,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            {
                let mut rng = thread_rng();
                let mapped = seeds.map_range_mut(.., vk::MemoryMapFlags::empty());
                for i in 0..seeds.size {
                    mapped.slice[i] = rng.gen();
                }
            }
            seeds
        };
        let bsdfs: TBuffer<GPUBsdf> = TBuffer::new(
            ctx,
            builder.bsdfs.len(),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        {
            let mapped = bsdfs.map_range_mut(.., vk::MemoryMapFlags::empty());
            mapped.slice.clone_from_slice(builder.bsdfs.as_slice());
        }
        let textures: TBuffer<GPUTexture> = TBuffer::new(
            ctx,
            builder.textures.len(),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        {
            let mapped = textures.map_range_mut(.., vk::MemoryMapFlags::empty());
            mapped.slice.clone_from_slice(builder.textures.as_slice());
        }
        let instances: TBuffer<GPUMeshInstance> = TBuffer::new(
            ctx,
            accel.instances.len(),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        {
            let mapped = instances.map_range_mut(.., vk::MemoryMapFlags::empty());
            mapped.slice.clone_from_slice(accel.instances.as_slice());
        }
        let sobolmat = TBuffer::new(
            ctx,
            SOBOL_BITS * SOBOL_MAX_DIMENSION,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        {
            assert!(sobolmat.size * std::mem::size_of::<u32>() == size_of_val(&SOBOL_MATRIX));
            let mapped = sobolmat.map_range_mut(.., vk::MemoryMapFlags::empty());
            mapped
                .slice
                .copy_from_slice(bytemuck::cast_slice(&SOBOL_MATRIX));
        }
        let sobol_states = TBuffer::new(
            ctx,
            npixels as usize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        {
            let mut rng = thread_rng();
            let mapped = sobol_states.map_range_mut(.., vk::MemoryMapFlags::empty());
            mapped.slice.iter_mut().for_each(|state:&mut GPUSobolSamplerState| {
                state.dimension = 0;
                state.sample = 0;
                state.rotation = rng.gen();
            });
        }
        Self {
            ctx: ctx.clone(),
            seeds,
            sobolmat,
            sobol_states,
            accel,
            bsdfs,
            textures,
            instances,
            image_textures: builder.image_textures,
            point_lights,
            area_lights,
            lights,
            mesh_area_distribution: area_dist,
            light_distribution: light_dist,
        }
    }
}
