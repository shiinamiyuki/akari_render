use crate::accel::bvh::BvhAccel;
use crate::accel::bvh::SweepSAHBuilder;
use crate::accel::qbvh::QBvhAccel;
use crate::bsdf::BsdfClosure;
use crate::bsdf::TransportMode;
use crate::distribution::Distribution1D;
use crate::texture::ShadingPoint;
use crate::util::binserde::Decode;
use crate::util::binserde::Encode;
use crate::*;
use crate::{accel::bvh, bsdf::Bsdf};

use akari_common::lazy_static::lazy_static;
use bumpalo::Bump;
use glam::BVec4A;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::exit;
use std::sync::Arc;
#[derive(Clone, Copy)]
pub struct SurfaceInteraction<'a> {
    pub shape: &'a dyn Shape,
    pub bsdf: Option<&'a dyn Bsdf>,
    pub triangle: ShadingTriangle<'a>,
    pub uv: Vec2,
    pub sp: ShadingPoint,
    pub ng: Vec3,
    pub ns: Vec3,
    pub t: f32,
    pub texcoord: Vec2,
}
impl<'a> SurfaceInteraction<'a> {
    pub fn evaluate_bsdf<'b>(
        &self,
        lambda: &mut SampledWavelengths,
        mode: TransportMode,
        arena: &'b Bump,
    ) -> Option<BsdfClosure<'b>>
    where
        'a: 'b,
    {
        if let Some(bsdf) = self.bsdf {
            let frame = Frame::from_normal(self.ns);
            Some(BsdfClosure {
                frame,
                closure: bsdf.evaluate(&self.sp, mode, lambda, arena),
            })
        } else {
            None
        }
    }
}
#[derive(Clone, Copy)]
pub struct SurfaceSample {
    pub p: Vec3,
    pub texcoords: Vec2,
    pub pdf: f32,
    pub ng: Vec3,
    pub ns: Vec3,
}
pub trait Shape: Sync + Send + AsAny {
    fn intersect(&self, ray: &Ray, invd: Option<Vec3A>) -> Option<RayHit>;
    fn occlude(&self, ray: &Ray, invd: Option<Vec3A>) -> bool;
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf>;
    fn shading_triangle<'a>(&'a self, prim_id: u32) -> ShadingTriangle<'a>;
    fn triangle(&self, prim_id: u32) -> Triangle;
    fn aabb(&self) -> Bounds3f;
    fn sample_surface(&self, u: Vec3) -> SurfaceSample;
    fn area(&self) -> f32;
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Triangle {
    pub vertices: [Vec3; 3],
}

impl Triangle {
    fn area(&self) -> f32 {
        let e0 = self.vertices[1] - self.vertices[0];
        let e1 = self.vertices[2] - self.vertices[0];
        0.5 * e0.cross(e1).length()
    }
    fn ng(&self) -> Vec3 {
        let e0 = self.vertices[1] - self.vertices[0];
        let e1 = self.vertices[2] - self.vertices[0];
        e0.cross(e1).normalize()
    }
}

impl Triangle {
    pub fn intersect4(&self, ray: &Ray4, active_mask: BVec4A) -> (BVec4A, Vec4, [Vec4; 2]) {
        let v0 = self.vertices[0];
        let v1 = self.vertices[1];
        let v2 = self.vertices[2];

        let e1 = v1 - v0;
        let e2 = v2 - v0;

        let e1 = [Vec4::splat(e1[0]), Vec4::splat(e1[1]), Vec4::splat(e1[2])];
        let e2 = [Vec4::splat(e2[0]), Vec4::splat(e2[1]), Vec4::splat(e2[2])];

        let hx = ray.d[1] * e2[2] - ray.d[2] * e2[1];
        let hy = ray.d[2] * e2[0] - ray.d[0] * e2[2];
        let hz = ray.d[0] * e2[1] - ray.d[1] * e2[0];

        let a = e1[0] * hx + e1[1] * hy + e1[2] * hz;
        let mask = a.cmpgt(Vec4::splat(-1e-7)) & a.cmplt(Vec4::splat(1e-7)) & active_mask;
        if !mask.any() {
            return (mask, Vec4::ZERO, [Vec4::ZERO; 2]);
        }

        let f = Vec4::ONE / a;
        let sx = ray.o[0] - v0[0];
        let sy = ray.o[1] - v0[1];
        let sz = ray.o[2] - v0[2];
        let u = f * (sx * hx + sy * hy + sz * hz);

        let qx = sy * e1[2] - sz * e1[1];
        let qy = sz * e1[0] - sx * e1[2];
        let qz = sx * e1[1] - sy * e1[0];

        let v = f * (ray.d[0] * qx + ray.d[1] * qy + ray.d[2] * qz);
        let t = f * (e2[0] * qx + e2[1] * qy + e2[2] * qz);

        let mask = mask
            & t.cmpge(ray.tmin)
            & t.cmplt(ray.tmax)
            & u.cmpge(Vec4::splat(0.0))
            & u.cmple(Vec4::splat(1.0))
            & v.cmpge(Vec4::splat(0.0))
            & (u + v).cmple(Vec4::splat(1.0));
        (mask, t, [u, v])
    }
    pub fn intersect(&self, ray: &Ray) -> Option<(f32, Vec2)> {
        let v0 = self.vertices[0];
        let v1 = self.vertices[1];
        let v2 = self.vertices[2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = ray.d.cross(edge2);
        let a = edge1.dot(h);
        if a > -1e-7 && a < 1e-7 {
            return None;
        }
        let f = 1.0 / a;
        let s = ray.o - v0;
        let u = f * s.dot(h);
        if !(u >= 0.0 && u <= 1.0){
            return None;
        }
        let q = s.cross(edge1);
        let v = f * ray.d.dot(q);
        if !(v >= 0.0 && u + v <= 1.0){
            return None;
        }
        let t = f * edge2.dot(q);
        // let hit = t >= ray.tmin && t < ray.tmax && u >= 0.0 && u <= 1.0 && v >= 0.0 && u + v <= 1.0;
        let hit = t >= ray.tmin && t < ray.tmax;
        if hit {
            Some((t, vec2(u, v)))
        } else {
            None
        }
    }
    #[allow(dead_code)]
    fn aabb(&self) -> Bounds3f {
        Bounds3f::default()
            .insert_point(self.vertices[0])
            .insert_point(self.vertices[1])
            .insert_point(self.vertices[2])
    }
}
#[derive(Copy, Clone)]
pub struct ShadingTriangle<'a> {
    pub vertices: [Vec3; 3],
    pub texcoords: [Vec2; 3],
    pub normals: [Vec3; 3],
    pub bsdf: Option<&'a dyn Bsdf>,
}
impl<'a> ShadingTriangle<'a> {
    pub fn texcoord(&self, uv: Vec2) -> Vec2 {
        lerp3(self.texcoords[0], self.texcoords[1], self.texcoords[2], uv)
    }
    pub fn ns(&self, uv: Vec2) -> Vec3 {
        lerp3(self.normals[0], self.normals[1], self.normals[2], uv).normalize()
    }
    pub fn p(&self, uv: Vec2) -> Vec3 {
        lerp3(self.vertices[0], self.vertices[1], self.vertices[2], uv)
    }
    pub fn ng(&self) -> Vec3 {
        Triangle {
            vertices: self.vertices,
        }
        .ng()
    }
}
impl<'a> Into<Triangle> for ShadingTriangle<'a> {
    fn into(self) -> Triangle {
        Triangle {
            vertices: self.vertices,
        }
    }
}

pub enum MeshBvh {
    Bvh(BvhAccel<TriangleMeshAccelData>),
    QBvh(QBvhAccel<TriangleMeshAccelData>),
}
impl MeshBvh {
    pub fn aabb(&self) -> Aabb {
        match self {
            MeshBvh::Bvh(x) => x.aabb,
            MeshBvh::QBvh(x) => x.aabb,
        }
    }
    pub fn data(&self) -> &TriangleMeshAccelData {
        match self {
            MeshBvh::Bvh(x) => &x.data,
            MeshBvh::QBvh(x) => &x.data,
        }
    }
    pub fn traverse<F: FnMut(&mut Ray, Vec3A, u32) -> bool>(
        &self,
        ray: Ray,
        inv_d: Option<Vec3A>,
        f: F,
    ) {
        match self {
            MeshBvh::Bvh(x) => x.traverse(ray, inv_d, f),
            MeshBvh::QBvh(x) => x.traverse(ray, inv_d, f),
        }
    }
}
pub struct TriangleMeshAccelData {
    pub mesh: Arc<TriangleMesh>,
}

impl bvh::BvhData for TriangleMeshAccelData {
    fn aabb(&self, idx: u32) -> Bounds3f {
        let face = self.mesh.indices[idx as usize];
        let v0: Vec3 = self.mesh.vertices[face[0] as usize].into();
        let v1: Vec3 = self.mesh.vertices[face[1] as usize].into();
        let v2: Vec3 = self.mesh.vertices[face[2] as usize].into();
        Bounds3f::default()
            .insert_point(v0)
            .insert_point(v1)
            .insert_point(v2)
    }
}

pub struct TriangleMeshInstance {
    pub accel: Arc<MeshBvh>,
    pub bsdf: Arc<dyn Bsdf>,
    pub area: f32,
    pub dist: Distribution1D,
}

pub struct MeshInstanceProxy {
    pub mesh: Arc<TriangleMesh>,
    pub bsdf: Arc<dyn Bsdf>,
}


impl Shape for MeshInstanceProxy {
    fn intersect(&self, _ray: &Ray, _: Option<Vec3A>) -> Option<RayHit> {
        panic!("shouldn't be called")
    }

    fn occlude(&self, _ray: &Ray, _: Option<Vec3A>) -> bool {
        panic!("shouldn't be called")
    }

    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        Some(self.bsdf.as_ref())
    }

    fn shading_triangle<'a>(&'a self, prim_id: u32) -> ShadingTriangle<'a> {
        ShadingTriangle {
            bsdf: self.bsdf(),
            ..self.mesh.shading_triangle(prim_id as usize)
        }
    }

    fn triangle(&self, prim_id: u32) -> Triangle {
        self.mesh.triangle(prim_id as usize)
    }

    fn aabb(&self) -> Bounds3f {
        panic!("shouldn't be called")
    }

    fn sample_surface(&self, _u: Vec3) -> SurfaceSample {
        panic!("shouldn't be called")
    }

    fn area(&self) -> f32 {
        self.mesh.area()
    }
}

impl Shape for TriangleMeshInstance {
    fn aabb(&self) -> Bounds3f {
        self.accel.aabb()
    }
    fn intersect(&self, ray: &Ray, inv_d: Option<Vec3A>) -> Option<RayHit> {
        let mut hit = None;
        self.accel.traverse(*ray, inv_d, |ray, inv_d, prim_id| {
            let triangle = self.triangle(prim_id);
            if let Some((t, uv)) = triangle.intersect(ray) {
                ray.tmax = t;
                hit = Some((t, uv, prim_id));
            }
            true
        });
        hit.map(|(t, uv, prim_id)| {
            let triangle = self.triangle(prim_id);
            RayHit {
                t,
                uv,
                prim_id,
                geom_id: 0,
                ng: triangle.ng(),
            }
        })
    }
    fn occlude(&self, ray: &Ray, inv_d: Option<Vec3A>) -> bool {
        let mut occluded = false;
        self.accel.traverse(*ray, inv_d, |ray, inv_d, prim_id| {
            let triangle = self.triangle(prim_id);
            if triangle.intersect(ray).is_some() {
                occluded = true;
                false
            } else {
                true
            }
        });
        occluded
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        Some(self.bsdf.as_ref())
    }
    fn area(&self) -> f32 {
        self.area
    }
    fn sample_surface(&self, u: Vec3) -> SurfaceSample {
        self.accel.data().mesh.sample_surface(u, &self.dist)
    }

    fn shading_triangle<'a>(&'a self, prim_id: u32) -> ShadingTriangle<'a> {
        ShadingTriangle {
            bsdf: self.bsdf(),
            ..self.accel.data().mesh.shading_triangle(prim_id as usize)
        }
    }

    fn triangle(&self, prim_id: u32) -> Triangle {
        self.accel.data().mesh.triangle(prim_id as usize)
    }
}
#[derive(Serialize, Deserialize, Clone)]
pub struct MeshFlags {
    pub reuse_texcoord_indices: bool,
    pub reuse_normal_indices: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TriangleMesh {
    pub name: String,
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub texcoords: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
    pub normal_indices: Vec<[u32; 3]>,
    pub texcoord_indices: Vec<[u32; 3]>,
}
impl Encode for TriangleMesh {
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.name.encode(writer)?;
        self.vertices.encode(writer)?;
        self.normals.encode(writer)?;
        self.texcoords.encode(writer)?;
        self.indices.encode(writer)?;
        self.normal_indices.encode(writer)?;
        self.texcoord_indices.encode(writer)?;
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
        let normal_indices = Decode::decode(reader)?;
        let texcoord_indices = Decode::decode(reader)?;
        Ok(Self {
            name,
            vertices,
            normal_indices,
            texcoord_indices,
            texcoords,
            indices,
            normals,
        })
    }
}
impl TriangleMesh {
    pub fn sample_surface(&self, u: Vec3, dist: &Distribution1D) -> SurfaceSample {
        let (idx, pdf_idx) = dist.sample_discrete(u[2]);
        // let face: UVec3 = self.indices[idx as usize].into();
        // let v0 = self.vertices[face[0] as usize].into();
        // let v1 = self.vertices[face[1] as usize].into();
        // let v2 = self.vertices[face[2] as usize].into();
        // let trig = Triangle {
        //     vertices: [v0, v1, v2],
        // };
        let trig = self.shading_triangle(idx);
        let uv = uniform_sample_triangle(vec2(u.x, u.y));
        let p = trig.p(uv);
        SurfaceSample {
            p,
            ng: trig.ng(),
            ns: trig.ns(uv),
            texcoords: trig.texcoord(uv),
            pdf: 1.0 / self.triangle(idx).area() * pdf_idx,
        }
    }
    pub fn triangle(&self, i: usize) -> Triangle {
        let face = self.indices[i];
        let v0 = self.vertices[face[0] as usize].into();
        let v1 = self.vertices[face[1] as usize].into();
        let v2 = self.vertices[face[2] as usize].into();
        Triangle {
            vertices: [v0, v1, v2],
        }
    }
    pub fn shading_triangle<'a>(&self, i: usize) -> ShadingTriangle<'a> {
        let face: UVec3 = self.indices[i].into();
        let v0: Vec3 = self.vertices[face[0] as usize].into();
        let v1: Vec3 = self.vertices[face[1] as usize].into();
        let v2: Vec3 = self.vertices[face[2] as usize].into();

        let ng = (v1 - v0).cross(v2 - v0).normalize();
        ShadingTriangle {
            vertices: [v0, v1, v2],
            texcoords: self.texcoords(i),
            bsdf: None,
            normals: self.normals(i, ng),
        }
    }
    pub fn area(&self) -> f32 {
        self.indices
            .iter()
            .enumerate()
            .map(|(i, _face)| self.triangle(i).area())
            .sum()
    }
    pub fn area_distribution(&self) -> Distribution1D {
        let f: Vec<_> = self
            .indices
            .iter()
            .enumerate()
            .map(|(i, _face)| self.triangle(i).area())
            .collect();
        Distribution1D::new(f.as_slice()).unwrap()
    }
    pub fn texcoords(&self, i: usize) -> [Vec2; 3] {
        if self.texcoords.is_empty() {
            let tc0 = vec2(0.0, 0.0);
            let tc1 = vec2(0.0, 1.0);
            let tc2 = vec2(1.0, 1.0);
            [tc0, tc1, tc2]
        } else {
            let face: UVec3 = self.texcoord_indices[i].into();
            let tc0 = self.texcoords[face[0] as usize].into();
            let tc1 = self.texcoords[face[1] as usize].into();
            let tc2 = self.texcoords[face[2] as usize].into();
            [tc0, tc1, tc2]
        }
    }
    pub fn normals(&self, i: usize, ng: Vec3) -> [Vec3; 3] {
        if self.normals.is_empty() {
            [ng; 3]
        } else {
            let face: UVec3 = self.normal_indices[i].into();
            let ns0 = self.normals[face[0] as usize].into();
            let ns1 = self.normals[face[1] as usize].into();
            let ns2 = self.normals[face[2] as usize].into();
            [ns0, ns1, ns2]
        }
    }
}

impl TriangleMesh {
    pub fn build_accel(self: &Arc<TriangleMesh>) -> BvhAccel<TriangleMeshAccelData> {
        let accel = SweepSAHBuilder::build(
            TriangleMeshAccelData { mesh: self.clone() },
            (0..self.indices.len() as u32).collect(),
        )
        .optimize_layout();
        accel
    }
    pub fn create_instance(
        bsdf: Arc<dyn Bsdf>,
        accel: Arc<MeshBvh>,
        mesh: Arc<TriangleMesh>,
    ) -> Arc<dyn Shape> {
        let instance = TriangleMeshInstance {
            accel,
            bsdf,
            area: mesh.area(),
            dist: mesh.area_distribution(),
        };
        Arc::new(instance)
    }
}
pub fn compute_normals(model: &mut TriangleMesh, angle: f32) {
    let angle = angle.to_radians();
    model.normals.clear();
    let mut face_normal_areas = vec![];
    let mut vertex_neighbors: HashMap<u32, Vec<u32>> = HashMap::new();
    for f in 0..model.indices.len() {
        let face = model.indices[f];
        for idx in face.iter() {
            if !vertex_neighbors.contains_key(idx) {
                vertex_neighbors.insert(*idx, vec![f as u32]);
            } else {
                vertex_neighbors.get_mut(idx).unwrap().push(f as u32);
            }
        }
        let triangle: Vec<Vec3> = face
            .iter()
            .map(|idx| model.vertices[*idx as usize].into())
            .collect();
        let edge0: Vec3 = triangle[1] - triangle[0];
        let edge1: Vec3 = triangle[2] - triangle[0];
        let ng = edge0.cross(edge1).normalize();
        let area = edge0.cross(edge1).length();
        face_normal_areas.push((ng, area));
    }
    model.normal_indices = (0..model.indices.len())
        .map(|i| {
            let i = i as u32;
            [3 * i, 3 * i + 1, 3 * i + 2]
        })
        .collect();
    model.normals = model
        .indices
        .iter()
        .enumerate()
        .flat_map(|(i, face)| {
            let (ng, area) = face_normal_areas[i];
            let vertex_neighbors = &vertex_neighbors;
            let face_normal_areas = &face_normal_areas;
            face.iter()
                .map(move |corner| -> [f32; 3] {
                    let neighbors = vertex_neighbors.get(corner).unwrap();
                    let mut w = area;
                    let mut ns = area * ng;
                    for n in neighbors {
                        let (n, a) = face_normal_areas[*n as usize];
                        if n.dot(ng).acos() < angle {
                            ns += n * w;
                            w += a;
                        }
                    }
                    (ns / w).normalize().into()
                })
                .into_iter()
        })
        .collect();
}

pub fn load_model(
    obj_file: &str,
    generate_normal: Option<f32>,
) -> (Vec<TriangleMesh>, Vec<tobj::Model>, Vec<tobj::Material>) {
    let (models, materials) = tobj::load_obj(
        &obj_file,
        &tobj::LoadOptions {
            triangulate: true,
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
            log::error!(
                "{} has model with empty name! all model must have a name",
                obj_file
            );
            exit(1);
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
        let mut indices = vec![];
        let mut normal_indices = vec![];
        let mut texcoord_indices = vec![];
        for f in 0..mesh.indices.len() / 3 {
            indices.push([
                mesh.indices[3 * f],
                mesh.indices[3 * f + 1],
                mesh.indices[3 * f + 2],
            ]);
        }
        if !mesh.normals.is_empty() && generate_normal.is_none() {
            for i in 0..mesh.normals.len() / 3 {
                normals.push([
                    mesh.normals[3 * i],
                    mesh.normals[3 * i + 1],
                    mesh.normals[3 * i + 2],
                ]);
            }
            for i in 0..mesh.normal_indices.len() / 3 {
                normal_indices.push([
                    mesh.normal_indices[3 * i],
                    mesh.normal_indices[3 * i + 1],
                    mesh.normal_indices[3 * i + 2],
                ]);
            }
        }
        if !mesh.texcoords.is_empty() {
            for i in 0..mesh.texcoords.len() / 2 {
                texcoords.push([mesh.texcoords[2 * i], mesh.texcoords[2 * i + 1]]);
            }
            for i in 0..mesh.texcoord_indices.len() / 3 {
                texcoord_indices.push([
                    mesh.texcoord_indices[3 * i],
                    mesh.texcoord_indices[3 * i + 1],
                    mesh.texcoord_indices[3 * i + 2],
                ]);
            }
        }
        let mut imported = TriangleMesh {
            name: m.name.clone(),
            vertices,
            normals,
            indices,
            texcoords,
            texcoord_indices,
            normal_indices,
        };
        if mesh.normals.is_empty() && generate_normal.is_some() {
            // todo!()
            println!("computing normals for {}", m.name);
            compute_normals(&mut imported, generate_normal.unwrap());
        }

        // let mut next_face = 0;
        // for f in 0..mesh.num_face_indices.len() {
        //     assert!(mesh.num_face_indices[f] == 3);
        //     let end = next_face + mesh.num_face_indices[f] as usize;
        //     let face_indices: Vec<_> = mesh.indices[next_face..end].iter().collect();
        //     println!("    face[{}] = {:?}", f, face_indices);
        //     next_face = end;
        // }
        imported_models.push(imported);
    }

    (imported_models, models, materials)
}
