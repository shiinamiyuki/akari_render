use std::collections::HashMap;
use std::sync::Arc;

use crate::geometry::ShadingTriangle;
use crate::heap::MegaHeap;
use crate::svm::ShaderRef;
use crate::util::binserde::*;
use crate::util::distribution::BindlessAliasTableVar;
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
#[derive(Clone, Copy, Debug)]
pub struct MeshInstanceHost {
    pub transform: AffineTransform,
    pub light: TagIndex,
    pub surface: ShaderRef,
    pub geom_id: u32,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_tangents: bool,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Value)]
pub struct MeshHeader {
    pub vertex_buf_idx: u32,
    pub index_buf_idx: u32,
    pub normal_buf_idx: u32,
    pub tangent_buf_idx: u32,
    pub uv_buf_idx: u32,
    pub area_sampler: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Value)]
pub struct MeshInstance {
    pub light: TagIndex,
    pub surface: ShaderRef,
    pub geom_id: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MeshAggregateHeader {
    pub mesh_instances: u32,
    pub mesh_transforms: u32,
    pub mesh_headers: u32,
}
pub struct MeshAggregate {
    pub heap: Arc<MegaHeap>,
    pub accel_meshes: Vec<rtx::Mesh>,
    pub accel: rtx::Accel,
    pub header: MeshAggregateHeader,
}
impl MeshAggregate {
    pub fn new(
        device: Device,
        heap: &Arc<MegaHeap>,
        meshes: &[&MeshBuffer],
        instances: &[MeshInstanceHost],
    ) -> Self {
        let mut accel_meshes = Vec::with_capacity(meshes.len());
        let accel = device.create_accel(AccelOption::default());
        let mut mesh_headers = vec![];
        for (i, mesh) in meshes.iter().enumerate() {
            let vertex_buf_idx = heap.bind_buffer(&mesh.vertices);
            let index_buf_idx = heap.bind_buffer(&mesh.indices);
            let normal_buf_idx = if mesh.has_normals {
                heap.bind_buffer(mesh.normals.as_ref().unwrap())
            } else {
                u32::MAX
            };
            let tangent_buf_idx = if mesh.has_tangents {
                heap.bind_buffer(mesh.tangents.as_ref().unwrap())
            } else {
                u32::MAX
            };

            let uv_buf_idx = if mesh.has_uvs {
                heap.bind_buffer(mesh.uvs.as_ref().unwrap())
            } else {
                u32::MAX
            };
            let accel_mesh = device.create_mesh(
                mesh.vertices.view(..),
                mesh.indices.view(..),
                AccelOption::default(),
            );
            accel_mesh.build(AccelBuildRequest::ForceBuild);
            accel_meshes.push(accel_mesh);
            let area_sampler = if let Some(at) = &mesh.area_sampler {
                let area_sampler = heap.bind_buffer(&at.0);
                let at1 = heap.bind_buffer(&at.1);
                assert_eq!(area_sampler + 1, at1);
                area_sampler
            } else {
                u32::MAX
            };
            mesh_headers.push(MeshHeader {
                vertex_buf_idx,
                index_buf_idx,
                normal_buf_idx,
                tangent_buf_idx,
                uv_buf_idx,
                area_sampler,
            });
        }
        let mesh_instances = device.create_buffer_from_fn(instances.len(), |i| {
            let inst = instances[i];
            MeshInstance {
                light: inst.light,
                surface: inst.surface,
                geom_id: inst.geom_id,
            }
        });
        let mesh_instance_idx = heap.bind_buffer(&mesh_instances);
        let mesh_headers = device.create_buffer_from_slice(&mesh_headers);
        let mesh_headers_idx = heap.bind_buffer(&mesh_headers);
        let mesh_transforms =
            device.create_buffer_from_fn(instances.len(), |i| instances[i].transform);
        let mesh_transform_idx = heap.bind_buffer(&mesh_transforms);
        for i in 0..instances.len() {
            let inst = &instances[i];
            let geom_id = inst.geom_id as usize;
            assert_eq!(inst.has_normals, meshes[geom_id].has_normals);
            assert_eq!(inst.has_uvs, meshes[geom_id].has_uvs);
            accel.push_mesh(&accel_meshes[geom_id], inst.transform.m, 255, true);
        }
        accel.build(AccelBuildRequest::ForceBuild);

        Self {
            heap: heap.clone(),
            accel_meshes,
            accel,
            header: MeshAggregateHeader {
                mesh_instances: mesh_instance_idx,
                mesh_transforms: mesh_transform_idx,
                mesh_headers: mesh_headers_idx,
            },
        }
    }

    #[tracked]
    pub fn mesh_vertices(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[f32; 3]> {
        self.heap.var().buffer(mesh_header.vertex_buf_idx)
    }
    #[tracked]
    pub fn mesh_indices(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[u32; 3]> {
        self.heap.var().buffer(mesh_header.index_buf_idx)
    }
    #[tracked]
    pub fn mesh_normals(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[f32; 3]> {
        self.heap.var().buffer(mesh_header.normal_buf_idx)
    }
    #[tracked]
    pub fn mesh_tangents(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[f32; 3]> {
        self.heap.var().buffer(mesh_header.tangent_buf_idx)
    }
    #[tracked]
    pub fn mesh_uvs(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[f32; 2]> {
        self.heap.var().buffer(mesh_header.uv_buf_idx)
    }
    #[tracked]
    pub fn mesh_area_samplers(&self, mesh_header: Expr<MeshHeader>) -> BindlessAliasTableVar {
        let b0 = self.heap.var().buffer(mesh_header.area_sampler);
        let b1 = self.heap.var().buffer(mesh_header.area_sampler + 1);
        BindlessAliasTableVar(b0, b1)
    }
    #[tracked]
    pub fn mesh_instances(&self) -> BindlessBufferVar<MeshInstance> {
        self.heap.var().buffer(self.header.mesh_instances)
    }
    #[tracked]
    pub fn mesh_instance_transforms(&self) -> BindlessBufferVar<AffineTransform> {
        self.heap.var().buffer(self.header.mesh_transforms)
    }
    #[tracked]
    pub fn shading_triangle(&self, inst_id: Expr<u32>, prim_id: Expr<u32>) -> ShadingTriangle {
        let inst: Expr<MeshInstance> = self.mesh_instances().read(inst_id);
        let geom_id = inst.geom_id;
        let geometry = self
            .heap
            .var()
            .buffer::<MeshHeader>(self.header.mesh_headers)
            .read(geom_id);
        let vertices = self.mesh_vertices(geometry);
        let indices = self.mesh_indices(geometry);
        let i: Expr<Uint3> = indices.read(prim_id).into();
        let transform = self.mesh_instance_transforms().read(inst_id);

        let v0 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.x)));
        let v1 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.y)));
        let v2 = transform.transform_point(Expr::<Float3>::from(vertices.read(i.z)));
        let prim_id3 = prim_id * 3;
        let (uv0, uv1, uv2) = if geometry.uv_buf_idx != u32::MAX {
            let uvs = self.mesh_uvs(geometry);
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
        let (n0, n1, n2) = if geometry.normal_buf_idx != u32::MAX {
            let normals = self.mesh_normals(geometry);
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
        let (t0, t1, t2) = if geometry.tangent_buf_idx != u32::MAX {
            let tangents = self.mesh_tangents(geometry);
            // let bitangent_signs = self.mesh_bitangent_signs.buffer(geom_id);
            let t0 = transform.transform_vector(Expr::<Float3>::from(tangents.read(prim_id3 + 0)));
            let t1 = transform.transform_vector(Expr::<Float3>::from(tangents.read(prim_id3 + 1)));
            let t2 = transform.transform_vector(Expr::<Float3>::from(tangents.read(prim_id3 + 2)));
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
            surface: inst.surface,
            ng,
        }
    }
}
