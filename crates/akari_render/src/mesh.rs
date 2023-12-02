use std::sync::Arc;

use crate::geometry::{map_to_sphere_host, Frame, FrameExpr};
use crate::heap::MegaHeap;
use crate::interaction::SurfaceInteraction;
use crate::svm::ShaderRef;
use crate::util::distribution::BindlessAliasTableVar;
use crate::*;
use crate::{geometry::AffineTransform, util::distribution::AliasTable};
use akari_common::luisa::lc_comment_lineno;
use luisa::rtx::*;

pub struct Mesh {
    pub vertices: Buffer<[f32; 3]>,
    pub normals: Option<Buffer<[f32; 3]>>,
    pub tangents: Option<Buffer<[f32; 3]>>,
    // pub bitangent_signs: Option<Buffer<u32>>,
    pub uvs: Option<Buffer<[f32; 2]>>,
    pub indices: Buffer<[u32; 3]>,
    pub material_slots: Buffer<u32>,
    pub area_sampler: Option<AliasTable>,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_tangents: bool,
}

pub struct MeshRef<'a> {
    pub vertices: &'a [[f32; 3]],
    pub normals: Option<&'a [[f32; 3]]>,
    pub tangents: Option<&'a [[f32; 3]]>,
    // pub bitangent_signs: &'a [u32],
    pub uvs: Option<&'a [[f32; 2]]>,
    pub indices: &'a [[u32; 3]],
    pub material_slots: &'a [u32],

    pub generated_tangents: Option<Vec<[f32; 3]>>,
    pub aabb: (glam::Vec3, glam::Vec3),
    pub inv_aabb_size: glam::Vec3,
}
impl<'a> mikktspace::Geometry for MeshRef<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len()
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.vertices[self.indices[face][vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        if let Some(normals) = &self.normals {
            normals[self.indices[face][vert] as usize]
        } else {
            let p0: glam::Vec3 = self.vertices[self.indices[face][0] as usize].into();
            let p1: glam::Vec3 = self.vertices[self.indices[face][1] as usize].into();
            let p2: glam::Vec3 = self.vertices[self.indices[face][2] as usize].into();
            let v0 = p1 - p0;
            let v1 = p2 - p0;
            v0.cross(v1).normalize().into()
        }
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        if let Some(uvs) = &self.uvs {
            uvs[self.indices[face][vert] as usize]
        } else {
            let p: glam::Vec3 = self.position(face, vert).into();
            let center = (self.aabb.0 + self.aabb.1) * 0.5;
            let p = (p + center) * self.inv_aabb_size;
            map_to_sphere_host(p).into()
        }
    }
    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let tangents = self.generated_tangents.as_mut().unwrap();
        tangents[self.indices[face][vert] as usize] = std::array::from_fn(|i| tangent[i] as f32);
    }
}
impl<'a> MeshRef<'a> {
    pub fn new(
        vertices: &'a [[f32; 3]],
        normals: Option<&'a [[f32; 3]]>,
        indices: &'a [[u32; 3]],
        materials: &'a [u32],
        uvs: Option<&'a [[f32; 2]]>,
        tangents: Option<&'a [[f32; 3]]>,
    ) -> Self {
        let mut aabb = (
            glam::Vec3::splat(std::f32::MAX),
            glam::Vec3::splat(std::f32::MIN),
        );
        for v in vertices {
            aabb.0 = aabb.0.min(glam::Vec3::from(*v));
            aabb.1 = aabb.1.max(glam::Vec3::from(*v));
        }
        let inv_aabb_size = 1.0 / (aabb.1 - aabb.0);
        Self {
            vertices,
            normals,
            tangents,
            uvs,
            indices,
            material_slots: materials,
            generated_tangents: None,
            aabb,
            inv_aabb_size,
        }
    }
    pub fn tangents(&self) -> &[[f32; 3]] {
        self.tangents
            .or(self.generated_tangents.as_ref().map(|t| t.as_slice()))
            .unwrap()
    }
    pub fn compute_tangents(&mut self) {
        self.generated_tangents = Some(vec![[0.0f32; 3]; self.vertices.len()]);
        if !mikktspace::generate_tangents(self) {
            log::warn!("failed to generate tangents for mesh");
        }
    }
}

impl Mesh {
    pub fn new(device: Device, args: MeshRef<'_>) -> Self {
        if let Some(normals) = &args.normals {
            assert_eq!(args.indices.len() * 3, normals.len());
        }
        if let Some(uvs) = &args.uvs {
            assert_eq!(args.indices.len() * 3, uvs.len());
        }
        if let Some(tangents) = &args.tangents {
            assert_eq!(args.indices.len() * 3, tangents.len());
        }
        assert!(args.material_slots.len() == 1 || args.material_slots.len() == args.indices.len());
        let vertices = device.create_buffer_from_slice(&args.vertices);
        let normals = args
            .normals
            .map(|normals| device.create_buffer_from_slice(normals));
        let tangents = args
            .tangents
            .map(|tangents| device.create_buffer_from_slice(tangents));
        // let bitangent_signs = if mesh.bitangent_signs.is_empty() {
        //     None
        // } else {
        //     Some(device.create_buffer_from_slice(&mesh.bitangent_signs))
        // };
        let indices = device.create_buffer_from_slice(&args.indices);
        let uvs = args.uvs.map(|uvs| device.create_buffer_from_slice(uvs));
        let materials = device.create_buffer_from_slice(args.material_slots);
        let m = Self {
            vertices,
            normals,
            tangents,
            // bitangent_signs,
            uvs,
            indices,
            area_sampler: None,
            has_normals: args.normals.is_some(),
            has_uvs: args.uvs.is_some(),
            has_tangents: args.tangents.is_some(),
            material_slots: materials,
        };
        // assert!(!m.has_uvs || m.has_tangents, "mesh has uvs but no tangents");
        m
    }
    pub fn build_area_sampler(&mut self, device: Device, areas: &[f32]) {
        self.area_sampler = Some(AliasTable::new(device, areas));
    }
}
#[repr(transparent)]
pub struct MeshInstanceFlags;
impl MeshInstanceFlags {
    pub const HAS_NORMALS: u32 = 1 << 0;
    pub const HAS_UVS: u32 = 1 << 1;
    pub const HAS_TANGENTS: u32 = 1 << 2;
    pub const HAS_MULTI_MATERIALS: u32 = 1 << 3;
}
#[repr(C)]
#[derive(Clone, Debug)]
pub struct MeshInstanceHost {
    pub transform: AffineTransform,
    pub light: TagIndex,
    pub materials: Vec<ShaderRef>,
    pub geom_id: u32,
    pub flags: u32,
}
impl MeshInstanceHost {
    pub fn has_normals(&self) -> bool {
        self.flags & MeshInstanceFlags::HAS_NORMALS != 0
    }
    pub fn has_uvs(&self) -> bool {
        self.flags & MeshInstanceFlags::HAS_UVS != 0
    }
    pub fn has_tangents(&self) -> bool {
        self.flags & MeshInstanceFlags::HAS_TANGENTS != 0
    }
    pub fn has_multi_materials(&self) -> bool {
        self.flags & MeshInstanceFlags::HAS_MULTI_MATERIALS != 0
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
pub struct MeshHeader {
    pub vertex_buf_idx: u32,
    pub index_buf_idx: u32,
    pub material_slots_buf_idx: u32,
    pub normal_buf_idx: u32,
    pub tangent_buf_idx: u32,
    pub uv_buf_idx: u32,
    pub area_sampler: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
pub struct MeshInstance {
    pub light: TagIndex,
    pub material_buffer_idx: u32,
    pub geom_id: u32,
    pub transform_det: f32,
    pub flags: u32,
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
        meshes: &[&Mesh],
        instances: &[MeshInstanceHost],
    ) -> Self {
        let mut accel_meshes = Vec::with_capacity(meshes.len());
        let accel = device.create_accel(AccelOption::default());
        let mut mesh_headers = vec![];
        for (_i, mesh) in meshes.iter().enumerate() {
            let vertex_buf_idx = heap.bind_buffer(&mesh.vertices);
            let index_buf_idx = heap.bind_buffer(&mesh.indices);
            let material_slots_buf_idx = heap.bind_buffer(&mesh.material_slots);
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
                material_slots_buf_idx,
            });
        }
        let mesh_instances = device.create_buffer_from_fn(instances.len(), |i| {
            let materials = device.create_buffer_from_slice(&instances[i].materials);
            let material_buf_index = heap.bind_buffer(&materials);
            let inst = &instances[i];
            let t: glam::Mat4 = inst.transform.m.into();
            MeshInstance {
                light: inst.light,
                material_buffer_idx: todo!(), //inst.surface,
                geom_id: inst.geom_id,
                transform_det: t.determinant(),
                flags: todo!(),
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
            assert_eq!(inst.has_normals(), meshes[geom_id].has_normals);
            assert_eq!(inst.has_uvs(), meshes[geom_id].has_uvs);
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

    #[tracked(crate = "luisa")]
    pub fn mesh_vertices(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[f32; 3]> {
        self.heap.buffer(mesh_header.vertex_buf_idx)
    }
    #[tracked(crate = "luisa")]
    pub fn mesh_indices(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[u32; 3]> {
        self.heap.buffer(mesh_header.index_buf_idx)
    }
    #[tracked(crate = "luisa")]
    pub fn mesh_material_slots(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<u32> {
        self.heap.buffer(mesh_header.material_slots_buf_idx)
    }
    #[tracked(crate = "luisa")]
    pub fn mesh_normals(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[f32; 3]> {
        self.heap.buffer(mesh_header.normal_buf_idx)
    }
    #[tracked(crate = "luisa")]
    pub fn mesh_tangents(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[f32; 3]> {
        self.heap.buffer(mesh_header.tangent_buf_idx)
    }
    #[tracked(crate = "luisa")]
    pub fn mesh_uvs(&self, mesh_header: Expr<MeshHeader>) -> BindlessBufferVar<[f32; 2]> {
        self.heap.buffer(mesh_header.uv_buf_idx)
    }
    #[tracked(crate = "luisa")]
    pub fn mesh_area_samplers(&self, mesh_header: Expr<MeshHeader>) -> BindlessAliasTableVar {
        let b0 = self.heap.buffer(mesh_header.area_sampler);
        let b1 = self.heap.buffer(mesh_header.area_sampler + 1);
        BindlessAliasTableVar(b0, b1)
    }
    #[tracked(crate = "luisa")]
    pub fn mesh_instances(&self) -> BindlessBufferVar<MeshInstance> {
        self.heap.buffer(self.header.mesh_instances)
    }
    #[tracked(crate = "luisa")]
    pub fn mesh_instance_transforms(&self) -> BindlessBufferVar<AffineTransform> {
        self.heap.buffer(self.header.mesh_transforms)
    }
    #[tracked(crate = "luisa")]
    pub fn surface_interaction(
        &self,
        inst_id: Expr<u32>,
        prim_id: Expr<u32>,
        bary: Expr<Float2>,
    ) -> SurfaceInteraction {
        let inst: Expr<MeshInstance> = self.mesh_instances().read(inst_id);
        let geom_id = inst.geom_id;
        let geometry = self
            .heap
            .buffer::<MeshHeader>(self.header.mesh_headers)
            .read(geom_id);
        let vertices = self.mesh_vertices(geometry);
        let indices = self.mesh_indices(geometry);
        let material_slots = self.mesh_material_slots(geometry);
        let material = self
            .heap
            .buffer::<ShaderRef>(inst.material_buffer_idx)
            .read(material_slots.read(prim_id));
        let i: Expr<Uint3> = indices.read(prim_id).into();

        let (area_local, p_local, ng_local) = {
            let v0 = Expr::<Float3>::from(vertices.read(i.x));
            let v1 = Expr::<Float3>::from(vertices.read(i.y));
            let v2 = Expr::<Float3>::from(vertices.read(i.z));
            let p = bary.interpolate(v0, v1, v2);
            let ng = (v1 - v0).cross(v2 - v0);
            let len = ng.length();
            let area = len * 0.5;
            let ng = ng / len;
            (area, p, ng)
        };
        let prim_id3 = prim_id * 3;
        let uv = if geometry.uv_buf_idx != u32::MAX {
            let uvs = self.mesh_uvs(geometry);
            let uv0: Expr<Float2> = uvs.read(prim_id3 + 0).into();
            let uv1: Expr<Float2> = uvs.read(prim_id3 + 1).into();
            let uv2: Expr<Float2> = uvs.read(prim_id3 + 2).into();
            bary.interpolate(uv0, uv1, uv2)
        } else {
            let uv0 = Float2::expr(0.0, 0.0);
            let uv1 = Float2::expr(1.0, 0.0);
            let uv2 = Float2::expr(0.0, 0.1);
            bary.interpolate(uv0, uv1, uv2)
        };

        let tt_local = {
            let t0 = Var::<Float3>::zeroed();
            let t1 = Var::<Float3>::zeroed();
            let t2 = Var::<Float3>::zeroed();
            let use_default = false.var();
            let t = Var::<Float3>::zeroed();
            if geometry.tangent_buf_idx != u32::MAX {
                let tangents = self.mesh_tangents(geometry);
                *t0 = Expr::<Float3>::from(tangents.read(prim_id3 + 0));
                *t1 = Expr::<Float3>::from(tangents.read(prim_id3 + 1));
                *t2 = Expr::<Float3>::from(tangents.read(prim_id3 + 2));
                let all_good = t0.is_finite().all() & t1.is_finite().all() & t2.is_finite().all();
                if !all_good {
                    *use_default = true;
                } else {
                    *t = bary.interpolate(**t0, **t1, **t2);
                }
            } else {
                *use_default = true;
            };
            if **use_default {
                let v0 = Expr::<Float3>::from(vertices.read(i.x));
                let v1 = Expr::<Float3>::from(vertices.read(i.y));
                let v2 = Expr::<Float3>::from(vertices.read(i.z));

                let t0 = (v1 - v0).normalize();
                let t1 = (v2 - v1).normalize();
                let t2 = (v0 - v2).normalize();
                *t = bary.interpolate(t0, t1, t2);
            };
            **t
        };

        let ns_local = if geometry.normal_buf_idx != u32::MAX {
            let normals = self.mesh_normals(geometry);
            let n0 = Expr::<Float3>::from(normals.read(prim_id3 + 0));
            let n1 = Expr::<Float3>::from(normals.read(prim_id3 + 1));
            let n2 = Expr::<Float3>::from(normals.read(prim_id3 + 2));
            bary.interpolate(n0, n1, n2)
        } else {
            ng_local
        };
        // apply transform
        lc_comment_lineno!(
            crate = [luisa],
            "MeshAggregate::surface_inteaction apply transform"
        );
        let (area, p, ng, ns, tt) = {
            let transform = self.mesh_instance_transforms().read(inst_id);
            // let close_to_identity = transform.close_to_identity;
            let m = transform.m;
            let t = m[3].xyz();
            let m = Mat3::from_elems_expr([m[0].xyz(), m[1].xyz(), m[2].xyz()]);
            let p = m * p_local + t;
            let tt = m * tt_local;

            let area = area_local * inst.transform_det / (m * ng_local).length();

            // let area = {
            //     let v0 = Expr::<Float3>::from(vertices.read(i.x));
            //     let v1 = Expr::<Float3>::from(vertices.read(i.y));
            //     let v2 = Expr::<Float3>::from(vertices.read(i.z));
            //     let e0 = m * (v1 - v0);
            //     let e1 = m * (v2 - v0);
            //     e0.cross(e1).length() * 0.5
            // };
            let m_inv_t = m.transpose().inverse();
            let ng = (m_inv_t * ng_local).normalize();
            let ns = (m_inv_t * ns_local).normalize();
            (area, p, ng, ns, tt)
        };
        let ss = ns.cross(tt);
        let frame = if ss.length_squared() > 0.0 {
            let ss = ss.normalize();
            let tt = ss.cross(ns).normalize();
            Frame::new_expr(ns, tt, ss)
        } else {
            FrameExpr::from_n(ns)
        };
        SurfaceInteraction {
            frame,
            p,
            ng,
            bary,
            uv,
            inst_id,
            prim_id,
            surface: material,
            prim_area: area,
            valid: true.expr(),
        }
    }
}
