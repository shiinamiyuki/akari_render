use std::collections::HashMap;

use crate::geometry::{ShadingTriangle, Triangle};
use crate::svm::ShaderRef;
use crate::util::binserde::*;
use crate::*;
use crate::{geometry::AffineTransform, util::distribution::AliasTable};
use luisa::resource::BufferHeap;
use luisa::rtx::*;
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
                let v0 = glam::Vec3::from(self.vertices[i[0] as usize]);
                let v1 = glam::Vec3::from(self.vertices[i[1] as usize]);
                let v2 = glam::Vec3::from(self.vertices[i[2] as usize]);
                (v1 - v0).cross(v2 - v0).length() / 2.0
            })
            .collect()
    }
}

pub struct MeshBuffer {
    pub vertices: Buffer<[f32; 3]>,
    pub normals: Option<Buffer<[f32; 3]>>,
    pub tangents: Option<Buffer<[f32; 3]>>,
    pub bitangent_signs: Option<Buffer<u32>>,
    pub uvs: Option<Buffer<[f32; 2]>>,
    pub indices: Buffer<[u32; 3]>,
    pub area_sampler: Option<AliasTable>,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_tangents: bool,
}
impl MeshBuffer {
    pub fn new(device: Device, mesh: &TriangleMesh) -> Self {
        if !mesh.normals.is_empty() {
            assert_eq!(mesh.indices.len() * 3, mesh.normals.len());
        }
        if !mesh.uvs.is_empty() {
            assert_eq!(mesh.indices.len() * 3, mesh.uvs.len());
        }
        if !mesh.tangents.is_empty() {
            assert_eq!(mesh.indices.len() * 3, mesh.tangents.len());
            assert_eq!(
                (mesh.indices.len() * 3 + 31) / 32,
                mesh.bitangent_signs.len()
            );
        }
        let vertices = device.create_buffer_from_slice(&mesh.vertices);
        let normals = if mesh.normals.is_empty() {
            None
        } else {
            Some(device.create_buffer_from_slice(&mesh.normals))
        };
        let tangents = if mesh.tangents.is_empty() {
            None
        } else {
            Some(device.create_buffer_from_slice(&mesh.tangents))
        };
        let bitangent_signs = if mesh.bitangent_signs.is_empty() {
            None
        } else {
            Some(device.create_buffer_from_slice(&mesh.bitangent_signs))
        };
        let indices = device.create_buffer_from_slice(&mesh.indices);
        let uvs = if mesh.uvs.is_empty() {
            None
        } else {
            Some(device.create_buffer_from_slice(&mesh.uvs))
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
    pub surface: ShaderRef,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_tangents: bool,
}
pub struct MeshAggregate {
    pub mesh_vertices: BufferHeap<[f32; 3]>,
    pub mesh_normals: BufferHeap<[f32; 3]>,
    pub mesh_tangents: BufferHeap<[f32; 3]>,
    pub mesh_bitangent_signs: BufferHeap<u32>,
    pub mesh_uvs: BufferHeap<[f32; 2]>,
    pub mesh_indices: BufferHeap<[u32; 3]>,
    pub mesh_instances: Buffer<MeshInstance>,
    pub mesh_area_samplers: BindlessArray,
    pub mesh_id_to_area_samplers: HashMap<u32, u32>,
    pub accel_meshes: Vec<rtx::Mesh>,
    pub accel: rtx::Accel,
}
impl MeshAggregate {
    pub fn new(device: Device, meshes: &[&MeshBuffer], instances: &[MeshInstance]) -> Self {
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
        let mut require_update_normals = false;
        let mut require_update_tangents = false;
        let mut require_update_uvs = false;
        let mut require_update_samplers = false;
        for (i, mesh) in meshes.iter().enumerate() {
            mesh_vertices.emplace_buffer_async(i, &mesh.vertices);
            mesh_indices.emplace_buffer_async(i, &mesh.indices);
            if mesh.has_normals {
                mesh_normals.emplace_buffer_async(i, mesh.normals.as_ref().unwrap());
                require_update_normals = true;
            }
            if mesh.has_tangents {
                mesh_tangents.emplace_buffer_async(i, mesh.tangents.as_ref().unwrap());
                mesh_bitangent_signs
                    .emplace_buffer_async(i, mesh.bitangent_signs.as_ref().unwrap());
                require_update_tangents = true;
            }
            if mesh.has_uvs {
                mesh_uvs.emplace_buffer_async(i, mesh.uvs.as_ref().unwrap());
                require_update_uvs = true;
            }
            let accel_mesh = device.create_mesh(
                mesh.vertices.view(..),
                mesh.indices.view(..),
                AccelOption::default(),
            );
            accel_mesh.build(AccelBuildRequest::ForceBuild);
            accel_meshes.push(accel_mesh);
            if let Some(at) = &mesh.area_sampler {
                mesh_area_samplers.emplace_buffer_async(at_cnt, &at.0);
                mesh_area_samplers.emplace_buffer_async(at_cnt + 1, &at.1);
                mesh_id_to_area_samplers.insert(i as u32, at_cnt as u32);
                at_cnt += 2;
                require_update_samplers = true;
            }
        }
        mesh_vertices.update();
        mesh_indices.update();
        if require_update_normals {
            mesh_normals.update();
        }
        if require_update_tangents {
            mesh_tangents.update();
            mesh_bitangent_signs.update();
        }
        if require_update_uvs {
            mesh_uvs.update();
        }
        if require_update_samplers {
            mesh_area_samplers.update();
        }
        for i in 0..instances.len() {
            let inst = &instances[i];
            let geom_id = inst.geom_id as usize;
            assert_eq!(inst.has_normals, meshes[geom_id].has_normals);
            assert_eq!(inst.has_uvs, meshes[geom_id].has_uvs);
            accel.push_mesh(&accel_meshes[geom_id], inst.transform.m, u32::MAX, true);
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
    #[tracked]
    pub fn triangle(&self, inst_id: Expr<u32>, prim_id: Expr<u32>) -> Triangle {
        let inst = self.mesh_instances.read(inst_id);
        let transform = inst.transform;
        let geom_id = inst.geom_id;
        let vertices = self.mesh_vertices.buffer(geom_id);
        let indices = self.mesh_indices.buffer(geom_id);
        let i: Expr<Uint3> = indices.read(prim_id).into();
        let v0 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.x)));
        let v1 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.y)));
        let v2 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.z)));
        Triangle { v0, v1, v2 }
    }
    #[tracked]
    pub fn shading_triangle(&self, inst_id: Expr<u32>, prim_id: Expr<u32>) -> ShadingTriangle {
        let inst = self.mesh_instances.read(inst_id);
        let geom_id = inst.geom_id;
        let vertices = self.mesh_vertices.buffer(geom_id);
        let indices = self.mesh_indices.buffer(geom_id);
        let i: Expr<Uint3> = indices.read(prim_id).into();
        let transform = inst.transform;
        let v0 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.x)));
        let v1 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.y)));
        let v2 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.z)));
        let prim_id3 = prim_id * 3;
        let (uv0, uv1, uv2) = if inst.has_uvs {
            let uvs = self.mesh_uvs.buffer(geom_id);
            let uv0 = uvs.read(prim_id3 + 0).into();
            let uv1 = uvs.read(prim_id3 + 1).into();
            let uv2 = uvs.read(prim_id3 + 2).into();
            (uv0, uv1, uv2)
        } else {
            let uv0 = Float2::expr(0.0, 0.0);
            let uv1 = Float2::expr(1.0, 0.0);
            let uv2 = Float2::expr(0.0, 0.1);
            (uv0, uv1, uv2)
        };
        let ng = (v1 - v0).cross(v2 - v0).normalize();
        let (n0, n1, n2) = if inst.has_normals {
            let normals = self.mesh_normals.buffer(geom_id);
            let n0 = transform.transform_normal(Expr::<Float3>::from(normals.read(prim_id3 + 0)));
            let n1 = transform.transform_normal(Expr::<Float3>::from(normals.read(prim_id3 + 1)));
            let n2 = transform.transform_normal(Expr::<Float3>::from(normals.read(prim_id3 + 2)));
            (n0, n1, n2)
        } else {
            (ng, ng, ng)
        };
        let make_default = || {
            let t0 = (v1 - v0).normalize();
            let t1 = (v2 - v1).normalize();
            let t2 = (v0 - v2).normalize();
            (t0, t1, t2)
        };
        let (t0, t1, t2) = if inst.has_tangents {
            let tangents = self.mesh_tangents.buffer(geom_id);
            // let bitangent_signs = self.mesh_bitangent_signs.buffer(geom_id);
            let t0 = transform.transform_vector(Expr::<Float3>::from(tangents.read(prim_id3 + 0)));
            let t1 = transform.transform_vector(Expr::<Float3>::from(tangents.read(prim_id3 + 1)));
            let t2 = transform.transform_vector(Expr::<Float3>::from(tangents.read(prim_id3 + 2)));

            // let get_sign = |i: u32| {
            //     let j = prim_id3 + i;
            //     let sign = bitangent_signs.read(j / 32);
            //     let sign = (sign >> (j % 32)) & 1;
            //     select(sign.eq(0), 1.0f32.expr(), -1.0f32.expr())
            // };
            // let s0 = get_sign(0u32);
            // let s1 = get_sign(1u32);
            // let s2 = get_sign(2u32);
            // let b0 = ng.cross(t0).normalize() * s0;
            // let b1 = ng.cross(t1).normalize() * s1;
            // let b2 = ng.cross(t2).normalize() * s2;
            let all_good = t0.is_finite().all() & t1.is_finite().all() & t2.is_finite().all();
            if !all_good {
                make_default()
            } else {
                (t0, t1, t2)
            }
        } else {
            make_default()
        };
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
            ng,
        }
    }
}
