use std::collections::HashMap;

use crate::geometry::{ShadingTriangle, Triangle};
use crate::util::binserde::*;
use crate::*;
use crate::{geometry::AffineTransform, util::alias_table::AliasTable};
use luisa::{AccelBuildRequest, AccelOption, BufferHeap};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TriangleMesh {
    pub name: String,
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 3]>,
    pub bitangent_signs: Vec<u32>, // bitmasks, 0 for 1, 1 for -1
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
}
impl Encode for TriangleMesh {
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.name.encode(writer)?;
        self.vertices.encode(writer)?;
        self.normals.encode(writer)?;
        self.tangents.encode(writer)?;
        self.bitangent_signs.encode(writer)?;
        self.uvs.encode(writer)?;
        self.indices.encode(writer)?;
        Ok(())
    }
}
impl Decode for TriangleMesh {
    fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        let name = Decode::decode(reader)?;
        let vertices = Decode::decode(reader)?;
        let normals = Decode::decode(reader)?;
        let tangents = Decode::decode(reader)?;
        let bitangent_signs = Decode::decode(reader)?;
        let uvs = Decode::decode(reader)?;
        let indices = Decode::decode(reader)?;
        Ok(Self {
            name,
            vertices,
            tangents,
            bitangent_signs,
            uvs,
            indices,
            normals,
        })
    }
}

impl TriangleMesh {
    pub fn areas(&self) -> Vec<f32> {
        self.indices
            .par_iter()
            .map(|i| {
                let v0 = glam::Vec3::from_array(self.vertices[i[0] as usize]);
                let v1 = glam::Vec3::from_array(self.vertices[i[1] as usize]);
                let v2 = glam::Vec3::from_array(self.vertices[i[2] as usize]);
                (v1 - v0).cross(v2 - v0).length() / 2.0
            })
            .collect()
    }
}
pub struct MeshBuffer {
    pub vertices: Buffer<PackedFloat3>,
    pub normals: Option<Buffer<PackedFloat3>>,
    pub tangents: Option<Buffer<PackedFloat3>>,
    pub bitangent_signs: Option<Buffer<u32>>,
    pub uvs: Option<Buffer<Float2>>,
    pub indices: Buffer<PackedUint3>,
    pub area_sampler: Option<AliasTable>,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_tangents: bool,
}
impl MeshBuffer {
    pub fn new(device: Device, mesh: &TriangleMesh) -> Self {
        if !mesh.normals.is_empty() {
            assert_eq!(mesh.indices.len(), mesh.normals.len());
        }
        if !mesh.uvs.is_empty() {
            assert_eq!(mesh.indices.len(), mesh.uvs.len());
        }
        if !mesh.tangents.is_empty() {
            assert_eq!(mesh.indices.len(), mesh.tangents.len());
            assert_eq!((mesh.indices.len() + 31) / 32, mesh.bitangent_signs.len());
        }
        let vertices = device.create_buffer_from_slice(unsafe {
            std::slice::from_raw_parts(
                mesh.vertices.as_ptr() as *const PackedFloat3,
                mesh.vertices.len(),
            )
        });
        let normals = if mesh.normals.is_empty() {
            None
        } else {
            Some(device.create_buffer_from_slice(unsafe {
                std::slice::from_raw_parts(
                    mesh.normals.as_ptr() as *const PackedFloat3,
                    mesh.normals.len(),
                )
            }))
        };
        let tangents = if mesh.tangents.is_empty() {
            None
        } else {
            Some(device.create_buffer_from_slice(unsafe {
                std::slice::from_raw_parts(
                    mesh.tangents.as_ptr() as *const PackedFloat3,
                    mesh.tangents.len(),
                )
            }))
        };
        let bitangent_signs = if mesh.bitangent_signs.is_empty() {
            None
        } else {
            Some(device.create_buffer_from_slice(unsafe {
                std::slice::from_raw_parts(
                    mesh.bitangent_signs.as_ptr() as *const u32,
                    mesh.bitangent_signs.len(),
                )
            }))
        };
        let indices = device.create_buffer_from_slice(unsafe {
            std::slice::from_raw_parts(
                mesh.indices.as_ptr() as *const PackedUint3,
                mesh.indices.len(),
            )
        });
        let uvs = if mesh.uvs.is_empty() {
            None
        } else {
            Some(device.create_buffer_from_slice(unsafe {
                std::slice::from_raw_parts(mesh.uvs.as_ptr() as *const Float2, mesh.uvs.len())
            }))
        };

        let m = Self {
            vertices,
            normals,
            tangents,
            bitangent_signs,
            uvs,
            indices,
            area_sampler: None,
            has_normals: !mesh.normals.is_empty(),
            has_uvs: !mesh.uvs.is_empty(),
            has_tangents: !mesh.tangents.is_empty(),
        };
        assert!(!m.has_uvs || m.has_tangents, "mesh has uvs but no tangents");
        m
    }
    pub fn build_area_sampler(&mut self, device: Device, areas: &[f32]) {
        self.area_sampler = Some(AliasTable::new(device, areas));
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Value)]
pub struct MeshInstance {
    pub geom_id: u32,
    pub transform: AffineTransform,
    pub light: TagIndex,
    pub surface: TagIndex,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_tangents: bool,
}
pub struct MeshAggregate {
    pub mesh_vertices: BufferHeap<PackedFloat3>,
    pub mesh_normals: BufferHeap<PackedFloat3>,
    pub mesh_tangents: BufferHeap<PackedFloat3>,
    pub mesh_bitangent_signs: BufferHeap<u32>,
    pub mesh_uvs: BufferHeap<Float2>,
    pub mesh_indices: BufferHeap<PackedUint3>,
    pub mesh_instances: Buffer<MeshInstance>,
    pub mesh_area_samplers: BindlessArray,
    pub mesh_id_to_area_samplers: HashMap<u32, u32>,
    pub accel_meshes: Vec<rtx::Mesh>,
    pub accel: rtx::Accel,
}
impl MeshAggregate {
    pub fn new(device: Device, meshes: &[&MeshBuffer], instances: &mut [MeshInstance]) -> Self {
        let count = meshes.len();
        let mesh_vertices = device.create_buffer_heap(count);
        let mesh_normals = device.create_buffer_heap(count);
        let mesh_tangents = device.create_buffer_heap(count);
        let mesh_uvs = device.create_buffer_heap(count);
        let mesh_bitangent_signs = device.create_buffer_heap(count);
        let mesh_indices = device.create_buffer_heap(count);
        let mesh_area_samplers = device.create_bindless_array(count * 2);
        let mut accel_meshes = Vec::with_capacity(meshes.len());
        let accel = device.create_accel(AccelOption::default());
        let mut at_cnt = 0;
        let mut mesh_id_to_area_samplers = HashMap::new();
        for (i, mesh) in meshes.iter().enumerate() {
            mesh_vertices.emplace_buffer(i, &mesh.vertices);
            mesh_indices.emplace_buffer(i, &mesh.indices);
            if mesh.has_normals {
                mesh_normals.emplace_buffer(i, mesh.normals.as_ref().unwrap());
            }
            if mesh.has_tangents {
                mesh_tangents.emplace_buffer(i, mesh.tangents.as_ref().unwrap());
                mesh_bitangent_signs.emplace_buffer(i, mesh.bitangent_signs.as_ref().unwrap());
            }
            if mesh.has_uvs {
                mesh_uvs.emplace_buffer(i, mesh.uvs.as_ref().unwrap());
            }
            let accel_mesh = device.create_mesh(
                mesh.vertices.view(..),
                mesh.indices.view(..),
                AccelOption::default(),
            );
            accel_mesh.build(AccelBuildRequest::ForceBuild);
            accel_meshes.push(accel_mesh);
            if let Some(at) = &mesh.area_sampler {
                mesh_area_samplers.emplace_buffer(at_cnt, &at.0);
                mesh_area_samplers.emplace_buffer(at_cnt + 1, &at.1);
                mesh_id_to_area_samplers.insert(i as u32, at_cnt as u32);
                at_cnt += 2;
            }
        }
        for i in 0..instances.len() {
            let inst = &mut instances[i];
            let geom_id = inst.geom_id as usize;
            inst.has_normals = meshes[geom_id].has_normals;
            inst.has_uvs = meshes[geom_id].has_uvs;
            accel.push_mesh(&accel_meshes[geom_id], inst.transform.m, u8::MAX, true);
        }
        accel.build(AccelBuildRequest::ForceBuild);
        let mesh_instances = device.create_buffer_from_slice(instances);
        Self {
            mesh_vertices,
            mesh_normals,
            mesh_tangents,
            mesh_bitangent_signs,
            mesh_uvs,
            mesh_indices,
            mesh_instances,
            accel_meshes,
            accel,
            mesh_area_samplers,
            mesh_id_to_area_samplers,
        }
    }
    pub fn triangle(&self, inst_id: Uint, prim_id: Uint) -> Triangle {
        let inst = self.mesh_instances.read(inst_id);
        let geom_id = inst.geom_id();
        let vertices = self.mesh_vertices.buffer(geom_id);
        let indices = self.mesh_indices.buffer(geom_id);
        let i = indices.read(prim_id);
        let v0 = vertices.read(i.x()).unpack();
        let v1 = vertices.read(i.y()).unpack();
        let v2 = vertices.read(i.z()).unpack();
        Triangle { v0, v1, v2 }
    }
    pub fn shading_triangle(&self, inst_id: Uint, prim_id: Uint) -> ShadingTriangle {
        let inst = self.mesh_instances.read(inst_id);
        let geom_id = inst.geom_id();
        let vertices = self.mesh_vertices.buffer(geom_id);
        let indices = self.mesh_indices.buffer(geom_id);
        let i = indices.read(prim_id);
        let v0 = vertices.read(i.x()).unpack();
        let v1 = vertices.read(i.y()).unpack();
        let v2 = vertices.read(i.z()).unpack();
        let prim_id3 = prim_id * 3;
        let (uv0, uv1, uv2) = if_!(
            inst.has_uvs(),
            {
                let uvs = self.mesh_uvs.buffer(geom_id);
                let uv0 = uvs.read(prim_id3 + 0);
                let uv1 = uvs.read(prim_id3 + 1);
                let uv2 = uvs.read(prim_id3 + 2);
                (uv0, uv1, uv2)
            },
            else,
            {
                let uv0 = make_float2(0.0, 0.0);
                let uv1 = make_float2(1.0, 0.0);
                let uv2 = make_float2(0.0, 0.1);
                (uv0, uv1, uv2)
            }
        );
        let ng = (v1 - v0).cross(v2 - v0).normalize();
        let (n0, n1, n2) = if_!(
            inst.has_normals(),
            {
                let normals = self.mesh_normals.buffer(geom_id);
                let n0 = normals.read(prim_id3 + 0).unpack();
                let n1 = normals.read(prim_id3 + 1).unpack();
                let n2 = normals.read(prim_id3 + 2).unpack();
                (n0, n1, n2)
            },
            else,
            { (ng, ng, ng) }
        );
        let (t0, t1, t2, b0, b1, b2) = if_!(
            inst.has_tangents(),
            {
                let tangents = self.mesh_tangents.buffer(geom_id);
                let bitangent_signs = self.mesh_bitangent_signs.buffer(geom_id);
                let t0 = tangents.read(prim_id3 + 0).unpack();
                let t1 = tangents.read(prim_id3 + 1).unpack();
                let t2 = tangents.read(prim_id3 + 2).unpack();
                let get_sign = |i: u32| {
                    let j = prim_id3 + i;
                    let sign = bitangent_signs.read(j / 32);
                    let sign = (sign >> (j % 32)) & 1;
                    select(sign.cmpeq(0), const_(1.0f32), const_(-1.0f32))
                };
                let s0 = get_sign(0u32);
                let s1 = get_sign(1u32);
                let s2 = get_sign(2u32);
                let b0 = ng.cross(t0) * s0;
                let b1 = ng.cross(t1) * s1;
                let b2 = ng.cross(t2) * s2;
                (t0, t1, t2, b0, b1, b2)
            },
            else,
            {
                let t0 = (v1 - v0).normalize();
                let t1 = (v2 - v1).normalize();
                let t2 = (v0 - v2).normalize();
                let b0 = ng.cross(t0);
                let b1 = ng.cross(t1);
                let b2 = ng.cross(t2);
                (t0, t1, t2, b0, b1, b2)
            }
        );
        ShadingTriangle {
            v0,
            v1,
            v2,
            uv0,
            uv1,
            uv2,
            n0,
            n1,
            n2,
            t0,
            t1,
            t2,
            b0,
            b1,
            b2,
            ng,
        }
    }
}
