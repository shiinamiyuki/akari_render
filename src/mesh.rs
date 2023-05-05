use std::collections::HashMap;

use crate::geometry::{ShadingTriangle, ShadingTriangleExpr, Triangle, TriangleExpr};
use crate::util::binserde::*;
use crate::*;
use crate::{geometry::AffineTransform, util::alias_table::AliasTable};
use luisa::{AccelBuildRequest, AccelOption};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TriangleMesh {
    pub name: String,
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub texcoords: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
}
impl Encode for TriangleMesh {
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.name.encode(writer)?;
        self.vertices.encode(writer)?;
        self.normals.encode(writer)?;
        self.texcoords.encode(writer)?;
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
        let texcoords = Decode::decode(reader)?;
        let indices = Decode::decode(reader)?;
        Ok(Self {
            name,
            vertices,
            texcoords,
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
    pub texcoords: Option<Buffer<PackedFloat2>>,
    pub indices: Buffer<PackedUint3>,
    pub area_sampler: Option<AliasTable>,
}
impl MeshBuffer {
    pub fn new(device: Device, mesh: &TriangleMesh) -> luisa::Result<Self> {
        let vertices = device.create_buffer_from_slice(unsafe {
            std::slice::from_raw_parts(
                mesh.vertices.as_ptr() as *const PackedFloat3,
                mesh.vertices.len(),
            )
        })?;
        let normals = if mesh.normals.len() > 0 {
            Some(device.create_buffer_from_slice(unsafe {
                std::slice::from_raw_parts(
                    mesh.normals.as_ptr() as *const PackedFloat3,
                    mesh.normals.len(),
                )
            })?)
        } else {
            None
        };
        let texcoords = if mesh.texcoords.len() > 0 {
            Some(device.create_buffer_from_slice(unsafe {
                std::slice::from_raw_parts(
                    mesh.texcoords.as_ptr() as *const PackedFloat2,
                    mesh.texcoords.len(),
                )
            })?)
        } else {
            None
        };
        let indices = device.create_buffer_from_slice(unsafe {
            std::slice::from_raw_parts(
                mesh.indices.as_ptr() as *const PackedUint3,
                mesh.indices.len(),
            )
        })?;
        Ok(Self {
            vertices,
            normals,
            texcoords,
            indices,
            area_sampler: None,
        })
    }
    pub fn build_area_sampler(&mut self, device: Device, areas: &[f32]) -> luisa::Result<()> {
        self.area_sampler = Some(AliasTable::new(device, areas)?);
        Ok(())
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Value)]
pub struct MeshInstance {
    pub geom_id: u32,
    pub transform: AffineTransform,
    pub normal_index: u32,
    pub texcoord_index: u32,
    pub emission_tex: TagIndex,
    pub surface: TagIndex,
}
impl MeshInstanceExpr {
    pub fn has_normal(&self) -> Bool {
        self.normal_index().cmpne(u32::MAX)
    }
    pub fn has_texcoord(&self) -> Bool {
        self.texcoord_index().cmpne(u32::MAX)
    }
}

pub struct MeshAggregate {
    pub mesh_vertices: BindlessArray,
    pub mesh_indices: BindlessArray,
    pub mesh_normals: BindlessArray,
    pub mesh_texcoords: BindlessArray,
    pub mesh_instances: Buffer<MeshInstance>,
    pub mesh_area_samplers: BindlessArray,
    pub mesh_id_to_area_samplers: HashMap<u32, u32>,
    pub accel_meshes: Vec<rtx::Mesh>,
    pub accel: rtx::Accel,
}
impl MeshAggregate {
    pub fn new(
        device: Device,
        meshes: &[&MeshBuffer],
        instances: &mut [MeshInstance],
    ) -> luisa::Result<Self> {
        let count = meshes.len();
        let mesh_vertices = device.create_bindless_array(count)?;
        let mesh_indices = device.create_bindless_array(count)?;
        let mesh_normals = device.create_bindless_array(count)?;
        let mesh_texcoords = device.create_bindless_array(count)?;
        let mesh_area_samplers = device.create_bindless_array(count * 2)?;
        let mut accel_meshes = Vec::with_capacity(meshes.len());
        let accel = device.create_accel(AccelOption::default())?;
        let mut mesh_id_to_normal_id = HashMap::new();
        let mut mesh_id_to_texcoord_id = HashMap::new();
        let mut at_cnt = 0;
        let mut mesh_id_to_area_samplers = HashMap::new();
        for (i, mesh) in meshes.iter().enumerate() {
            mesh_vertices.emplace_buffer_async(i, &mesh.vertices);
            mesh_indices.emplace_buffer_async(i, &mesh.indices);
            if let Some(normals) = &mesh.normals {
                mesh_normals.emplace_buffer_async(mesh_id_to_normal_id.len(), normals);
                mesh_id_to_normal_id.insert(i, mesh_id_to_normal_id.len() as u32);
            }
            if let Some(texcoords) = &mesh.texcoords {
                mesh_texcoords.emplace_buffer_async(mesh_id_to_texcoord_id.len(), texcoords);
                mesh_id_to_texcoord_id.insert(i, mesh_id_to_texcoord_id.len() as u32);
            }
            let accel_mesh = device.create_mesh(
                mesh.vertices.view(..),
                mesh.indices.view(..),
                AccelOption::default(),
            )?;
            accel_mesh.build(AccelBuildRequest::ForceBuild);
            accel_meshes.push(accel_mesh);
            if let Some(at) = &mesh.area_sampler {
                mesh_area_samplers.emplace_buffer_async(at_cnt, &at.0);
                mesh_area_samplers.emplace_buffer_async(at_cnt + 1, &at.1);
                mesh_id_to_area_samplers.insert(i as u32, at_cnt as u32);
                at_cnt += 2;
            }
        }
        for i in 0..instances.len() {
            let inst = &mut instances[i];
            let geom_id = inst.geom_id as usize;
            inst.normal_index = *mesh_id_to_normal_id.get(&geom_id).unwrap_or(&u32::MAX);
            inst.texcoord_index = *mesh_id_to_texcoord_id.get(&geom_id).unwrap_or(&u32::MAX);
            accel.push_mesh(&accel_meshes[geom_id], inst.transform.m, u8::MAX, true);
        }
        accel.build(AccelBuildRequest::ForceBuild);
        let mesh_instances = device.create_buffer_from_slice(instances)?;
        mesh_vertices.update();
        mesh_normals.update();
        mesh_texcoords.update();
        mesh_indices.update();
        mesh_area_samplers.update();
        Ok(Self {
            mesh_vertices,
            mesh_indices,
            mesh_normals,
            mesh_texcoords,
            mesh_instances,
            accel_meshes,
            accel,
            mesh_area_samplers,
            mesh_id_to_area_samplers,
        })
    }
    pub fn triangle(&self, inst_id: Uint, prim_id: Uint) -> Expr<Triangle> {
        let inst = self.mesh_instances.var().read(inst_id);
        let geom_id = inst.geom_id();
        let vertices = self.mesh_vertices.var().buffer::<Float3>(geom_id);
        let indices = self.mesh_indices.var().buffer::<Uint3>(geom_id);
        let i = indices.read(prim_id);
        let v0 = vertices.read(i.x());
        let v1 = vertices.read(i.y());
        let v2 = vertices.read(i.z());
        TriangleExpr::new(v0, v1, v2)
    }
    pub fn shading_triangle(&self, inst_id: Uint, prim_id: Uint) -> Expr<ShadingTriangle> {
        let inst = self.mesh_instances.var().read(inst_id);
        let geom_id = inst.geom_id();
        let vertices = self.mesh_vertices.var().buffer::<Float3>(geom_id);
        let indices = self.mesh_indices.var().buffer::<Uint3>(geom_id);
        let i = indices.read(prim_id);
        let v0 = vertices.read(i.x());
        let v1 = vertices.read(i.y());
        let v2 = vertices.read(i.z());
        let (tc0, tc1, tc2) = if_!(inst.has_texcoord(), {
            let texcoords = self.mesh_texcoords.var().buffer::<Float2>(inst.texcoord_index());
            let indices = self.mesh_texcoords.var().buffer::<Uint3>(inst.texcoord_index());
            let i = indices.read(prim_id);
            let tc0 = texcoords.read(i.x());
            let tc1 = texcoords.read(i.y());
            let tc2 = texcoords.read(i.z());
            (tc0, tc1, tc2)
        }, else {
            let tc0 = make_float2(0.0, 0.0);
            let tc1 = make_float2(1.0, 0.0);
            let tc2 = make_float2(0.0, 0.1);
            (tc0, tc1, tc2)
        });
        let ng = (v2 - v0).cross(v1 - v0).normalize();
        let (n0, n1, n2) = if_!(inst.has_normal(), {
            let normals = self.mesh_normals.var().buffer::<Float3>(inst.normal_index());
            let indices = self.mesh_normals.var().buffer::<Uint3>(inst.normal_index());
            let i = indices.read(prim_id);
            let n0 = normals.read(i.x());
            let n1 = normals.read(i.y());
            let n2 = normals.read(i.z());
            (n0, n1, n2)
        }, else {
            (ng, ng, ng)
        });
        ShadingTriangleExpr::new(v0, v1, v2, tc0, tc1, tc2, n0, n1, n2, ng)
    }
}
