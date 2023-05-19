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
#[derive(Copy, Clone, Value, Debug)]
#[repr(C)]
pub struct Vertex {
    pub position: PackedFloat3,
    pub normal: PackedFloat3,
    pub texcoord: Float2,
}
pub struct MeshBuffer {
    pub vertices: Buffer<Vertex>,
    pub indices: Buffer<PackedUint3>,
    pub area_sampler: Option<AliasTable>,
    pub has_normals: bool,
    pub has_texcoords: bool,
}
impl MeshBuffer {
    pub fn new(device: Device, mesh: &TriangleMesh) ->Self {
        let mut vertices = vec![];
        if !mesh.normals.is_empty() {
            assert_eq!(mesh.vertices.len(), mesh.normals.len());
        }
        if !mesh.texcoords.is_empty() {
            assert_eq!(mesh.vertices.len(), mesh.texcoords.len());
        }
        for i in 0..mesh.vertices.len() {
            let position: glam::Vec3 = mesh.vertices[i].into();
            let normal: glam::Vec3 = if mesh.normals.is_empty() {
                glam::Vec3::ZERO
            } else {
                mesh.normals[i].into()
            };
            let texcoords: glam::Vec2 = if mesh.texcoords.is_empty() {
                glam::Vec2::ZERO
            } else {
                mesh.texcoords[i].into()
            };
            let v = Vertex {
                position: position.into(),
                normal: normal.into(),
                texcoord: texcoords.into(),
            };
            vertices.push(v);
        }
        let vertices = device.create_buffer_from_slice(&vertices);
        let indices = device.create_buffer_from_slice(unsafe {
            std::slice::from_raw_parts(
                mesh.indices.as_ptr() as *const PackedUint3,
                mesh.indices.len(),
            )
        });
        Self {
            vertices,
            indices,
            area_sampler: None,
            has_normals: !mesh.normals.is_empty(),
            has_texcoords: !mesh.texcoords.is_empty(),
        }
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
    pub has_normals: bool,
    pub has_texcoords: bool,
    pub light: TagIndex,
    pub surface: TagIndex,
}
pub struct MeshAggregate {
    pub mesh_vertices: BindlessArray,
    pub mesh_indices: BindlessArray,
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
    ) -> Self {
        let count = meshes.len();
        let mesh_vertices = device.create_bindless_array(count);
        let mesh_indices = device.create_bindless_array(count);
        let mesh_normals = device.create_bindless_array(count);
        let mesh_texcoords = device.create_bindless_array(count);
        let mesh_area_samplers = device.create_bindless_array(count * 2);
        let mut accel_meshes = Vec::with_capacity(meshes.len());
        let accel = device.create_accel(AccelOption::default());
        let mut at_cnt = 0;
        let mut mesh_id_to_area_samplers = HashMap::new();
        for (i, mesh) in meshes.iter().enumerate() {
            mesh_vertices.emplace_buffer_async(i, &mesh.vertices);
            mesh_indices.emplace_buffer_async(i, &mesh.indices);
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
            }
        }
        for i in 0..instances.len() {
            let inst = &mut instances[i];
            let geom_id = inst.geom_id as usize;
            inst.has_normals = meshes[geom_id].has_normals;
            inst.has_texcoords = meshes[geom_id].has_texcoords;
            accel.push_mesh(&accel_meshes[geom_id], inst.transform.m, u8::MAX, true);
        }
        accel.build(AccelBuildRequest::ForceBuild);
        let mesh_instances = device.create_buffer_from_slice(instances);
        mesh_vertices.update();
        mesh_normals.update();
        mesh_texcoords.update();
        mesh_indices.update();
        mesh_area_samplers.update();
        Self {
            mesh_vertices,
            mesh_indices,
            mesh_instances,
            accel_meshes,
            accel,
            mesh_area_samplers,
            mesh_id_to_area_samplers,
        }
    }
    pub fn triangle(&self, inst_id: Uint, prim_id: Uint) -> Expr<Triangle> {
        let inst = self.mesh_instances.var().read(inst_id);
        let geom_id = inst.geom_id();
        let vertices = self.mesh_vertices.var().buffer::<Vertex>(geom_id);
        let indices = self.mesh_indices.var().buffer::<PackedUint3>(geom_id);
        let i = indices.read(prim_id);
        let v0 = vertices.read(i.x()).position().unpack();
        let v1 = vertices.read(i.y()).position().unpack();
        let v2 = vertices.read(i.z()).position().unpack();
        TriangleExpr::new(v0, v1, v2)
    }
    pub fn shading_triangle(&self, inst_id: Uint, prim_id: Uint) -> Expr<ShadingTriangle> {
        let inst = self.mesh_instances.var().read(inst_id);
        let geom_id = inst.geom_id();
        let vertices = self.mesh_vertices.var().buffer::<Vertex>(geom_id);
        let indices = self.mesh_indices.var().buffer::<PackedUint3>(geom_id);
        let i = indices.read(prim_id);
        let v0 = vertices.read(i.x());
        let v1 = vertices.read(i.y());
        let v2 = vertices.read(i.z());
        let (tc0, tc1, tc2) = if_!(inst.has_texcoords(), {
            (v0.texcoord(), v1.texcoord(), v2.texcoord())
        }, else {
            let tc0 = make_float2(0.0, 0.0);
            let tc1 = make_float2(1.0, 0.0);
            let tc2 = make_float2(0.0, 0.1);
            (tc0, tc1, tc2)
        });
        let ng = (v1.position().unpack() - v0.position().unpack())
            .cross(v2.position().unpack() - v0.position().unpack())
            .normalize();
        let (n0, n1, n2) = if_!(inst.has_normals(), {
            let n0 = v0.normal().unpack();
            let n1 = v1.normal().unpack();
            let n2 = v2.normal().unpack();
            (n0, n1, n2)
        }, else {
            (ng, ng, ng)
        });
        ShadingTriangleExpr::new(
            v0.position(),
            v1.position(),
            v2.position(),
            tc0,
            tc1,
            tc2,
            n0,
            n1,
            n2,
            ng,
        )
    }
}

pub fn load_model(
    obj_file: &str,
    _generate_normal: Option<f32>,
) -> (Vec<TriangleMesh>, Vec<tobj::Model>, Vec<tobj::Material>) {
    let (models, materials) = tobj::load_obj(
        &obj_file,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .expect("Failed to load file");
    let materials = materials.unwrap();
    let mut imported_models = vec![];
    // println!("# of models: {}", models.len());
    // println!("# of materials: {}", materials.len());
    for (_i, m) in models.iter().enumerate() {
        let mesh = &m.mesh;
        if m.name.is_empty() {
            panic!(
                "{} has model with empty name! all model must have a name",
                obj_file
            );
        }
        // println!("model[{}].name = \'{}\'", i, m.name);
        // println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

        // println!(
        //     "Size of model[{}].num_face_indices: {}",
        //     i,
        //     mesh.num_face_indices.len()
        // );
        let mut vertices = vec![];
        let mut normals = vec![];
        let mut texcoords = vec![];
        // let mut indices = vec![];
        assert!(mesh.positions.len() % 3 == 0);

        for v in 0..mesh.positions.len() / 3 {
            vertices.push([
                mesh.positions[3 * v],
                mesh.positions[3 * v + 1],
                mesh.positions[3 * v + 2],
            ]);
        }
        if !mesh.normals.is_empty() {
            assert_eq!(mesh.normals.len() % 3, 0);
            assert_eq!(mesh.normals.len(), mesh.positions.len());
            for v in 0..mesh.normals.len() / 3 {
                normals.push([
                    mesh.normals[3 * v],
                    mesh.normals[3 * v + 1],
                    mesh.normals[3 * v + 2],
                ]);
            }
        }
        if !mesh.texcoords.is_empty() {
            assert_eq!(mesh.texcoords.len() % 2, 0);
            assert_eq!(mesh.texcoords.len() / 2, mesh.positions.len() / 3);
            for v in 0..mesh.texcoords.len() / 2 {
                texcoords.push([mesh.texcoords[2 * v], mesh.texcoords[2 * v + 1]]);
            }
        }
        let mut indices = vec![];
        for f in 0..mesh.indices.len() / 3 {
            indices.push([
                mesh.indices[3 * f],
                mesh.indices[3 * f + 1],
                mesh.indices[3 * f + 2],
            ]);
        }
        let imported = TriangleMesh {
            name: m.name.clone(),
            vertices,
            normals,
            indices,
            texcoords,
        };
        imported_models.push(imported);
    }

    (imported_models, models, materials)
}
