use crate::accel::bvh::BVHAccelerator;
use crate::accel::bvh::SweepSAHBuilder;
use crate::distribution::Distribution1D;
use crate::*;
use crate::{accel::bvh, bsdf::Bsdf};

use serde::{Deserialize, Serialize};
use std::process::exit;
use std::sync::Arc;
extern crate tobj;

pub struct SurfaceSample {
    pub p: Vec3,
    pub texcoords: Vec2,
    pub pdf: Float,
    pub ng: Vec3,
    pub ns: Vec3,
}
pub trait Shape: Sync + Send + Base {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>>;
    fn occlude(&self, ray: &Ray) -> bool;
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf>;
    fn aabb(&self) -> Bounds3f;
    fn sample_surface(&self, u: &Vec3) -> SurfaceSample;
    fn area(&self) -> Float;
    fn children(&self) -> Option<Vec<Arc<dyn Shape>>> {
        None
    }
}

#[derive(Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: Float,
    pub bsdf: Arc<dyn Bsdf>,
}
impl_base!(Sphere);
impl Shape for Sphere {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>> {
        let oc = ray.o - self.center;
        let a = glm::dot(&ray.d, &ray.d);
        let b = 2.0 * glm::dot(&ray.d, &oc);
        let c = glm::dot(&oc, &oc) - self.radius * self.radius;
        let delta = b * b - 4.0 * a * c;
        if delta < 0.0 {
            return None;
        }
        let t1 = (-b - delta.sqrt()) / (2.0 * a);
        if t1 >= ray.tmin && t1 < ray.tmax {
            let p = ray.at(t1);
            let n = glm::normalize(&(p - self.center));
            let uv = dir_to_uv(&n);
            return Some(Intersection {
                shape: Some(self),
                t: t1,
                uv,
                texcoords: uv,
                ng: n,
                ns: n,
                prim_id: 0,
            });
        }
        let t2 = (-b + delta.sqrt()) / (2.0 * a);
        if t2 >= ray.tmin && t2 < ray.tmax {
            let p = ray.at(t2);
            let n = glm::normalize(&(p - self.center));
            let uv = dir_to_uv(&n);
            return Some(Intersection {
                shape: Some(self),
                t: t2,
                uv,
                texcoords: uv,
                ng: n,
                ns: n,
                prim_id: 0,
            });
        }
        None
    }
    fn occlude(&self, ray: &Ray) -> bool {
        let oc = ray.o - self.center;
        let a = glm::dot(&ray.d, &ray.d);
        let b = 2.0 * glm::dot(&ray.d, &oc);
        let c = glm::dot(&oc, &oc) - self.radius * self.radius;
        let delta = b * b - 4.0 * a * c;
        if delta < 0.0 {
            return false;
        }
        let t1 = (-b - delta.sqrt()) / (2.0 * a);
        if t1 >= ray.tmin && t1 < ray.tmax {
            return true;
        }
        let t2 = (-b + delta.sqrt()) / (2.0 * a);
        if t2 >= ray.tmin && t2 < ray.tmax {
            return true;
        }
        false
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        Some(&*self.bsdf)
    }
    fn aabb(&self) -> Bounds3f {
        Bounds3f {
            min: self.center - vec3(self.radius, self.radius, self.radius),
            max: self.center + vec3(self.radius, self.radius, self.radius),
        }
    }
    fn sample_surface(&self, u: &Vec3) -> SurfaceSample {
        let v = uniform_sphere_sampling(&vec2(u.x, u.y));
        SurfaceSample {
            p: self.center + v * self.radius,
            pdf: uniform_sphere_pdf(),
            texcoords: dir_to_uv(&v),
            ng: v,
            ns: v,
        }
    }
    fn area(&self) -> Float {
        4.0 * PI * self.radius
    }
}
#[derive(Clone)]
pub struct Triangle {
    pub vertices: [Vec3; 3],
    // pub texcoords: [Vec2; 3],
    // pub bsdf: Option<Arc<dyn Bsdf>>,
}

impl Triangle {
    fn area(&self) -> Float {
        let e0 = self.vertices[1] - self.vertices[0];
        let e1 = self.vertices[2] - self.vertices[0];
        glm::cross(&e0, &e1).norm() * 0.5
    }
    fn ng(&self) -> Vec3 {
        let e0 = self.vertices[1] - self.vertices[0];
        let e1 = self.vertices[2] - self.vertices[0];
        glm::normalize(&glm::cross(&e0, &e1))
    }
}

impl Triangle {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<(Float, Vec2)> {
        let v0 = &self.vertices[0].cast::<Float>();
        let v1 = &self.vertices[1].cast::<Float>();
        let v2 = &self.vertices[2].cast::<Float>();

        // float a,f,u,v;
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let h = glm::cross(&ray.d, &edge2); // rayVector.crossProduct(edge2);
        let a = glm::dot(&edge1, &h); //edge1.dotProduct(h);
        if a > -1e-7 && a < 1e-7 {
            return None; // This ray is parallel to this triangle.
        }
        let f = 1.0 / a;
        let s = ray.o - v0;
        let u = f * glm::dot(&s, &h); //s.dotProduct(h);
        if u < 0.0 || u > 1.0 {
            return None;
        }
        let q = glm::cross(&s, &edge1); //s.crossProduct(edge1);
        let v = f * glm::dot(&ray.d, &q); //rayVector.dotProduct(q);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        // At this stage we can compute t to find out where the intersection point is on the line.
        let t = f * glm::dot(&edge2, &q); //edge2.dotProduct(q);
        if t >= ray.tmin && t < ray.tmax
        // ray intersection
        {
            Some((t, vec2(u, v)))
        } else {
            None
        }
    }

    fn occlude(&self, ray: &Ray) -> bool {
        if let Some(_) = self.intersect(ray) {
            true
        } else {
            false
        }
    }
    fn aabb(&self) -> Bounds3f {
        Bounds3f::default()
            .insert_point(&self.vertices[0])
            .insert_point(&self.vertices[1])
            .insert_point(&self.vertices[2])
    }
}
pub struct TriangleMeshAccelData {
    pub mesh: Arc<TriangleMesh>,
}

impl TriangleMeshAccelData {
    fn get_tc(&self, face: &glm::UVec3) -> (Vec2, Vec2, Vec2) {
        self.mesh.get_tc(face)
    }
}
impl bvh::BVHData for TriangleMeshAccelData {
    fn aabb(&self, idx: u32) -> Bounds3f {
        let face = self.mesh.indices[idx as usize];
        let v0 = &self.mesh.vertices[face[0] as usize].cast::<Float>();
        let v1 = &self.mesh.vertices[face[1] as usize].cast::<Float>();
        let v2 = &self.mesh.vertices[face[2] as usize].cast::<Float>();
        Bounds3f::default()
            .insert_point(&v0)
            .insert_point(&v1)
            .insert_point(&v2)
    }
    fn intersect<'a>(&'a self, idx: u32, ray: &Ray) -> Option<Intersection<'a>> {
        let face = self.mesh.indices[idx as usize];
        let v0 = self.mesh.vertices[face[0] as usize].cast::<Float>();
        let v1 = self.mesh.vertices[face[1] as usize].cast::<Float>();
        let v2 = self.mesh.vertices[face[2] as usize].cast::<Float>();

        let trig = Triangle {
            vertices: [v0, v1, v2],
        };
        let t_uv = trig.intersect(ray);
        if let Some(t_uv) = t_uv {
            //let (tc0, tc1, tc2) = self.get_tc(&face);
            // let ng = trig.ng();
            Some(Intersection {
                shape: None,
                uv: t_uv.1,
                texcoords: vec2(0.0, 0.0), //lerp3(&tc0, &tc1, &tc2, &t_uv.1),
                ng: vec3(0.0, 0.0, 0.0),
                ns: vec3(0.0, 0.0, 0.0),
                t: t_uv.0,
                prim_id: idx,
            })
        } else {
            None
        }
    }
    fn bsdf<'a>(&'a self, _: u32) -> Option<&'a dyn Bsdf> {
        panic!("dont call this")
    }
    fn occlude(&self, idx: u32, ray: &Ray) -> bool {
        let face = self.mesh.indices[idx as usize];
        let v0 = self.mesh.vertices[face[0] as usize].cast::<Float>();
        let v1 = self.mesh.vertices[face[1] as usize].cast::<Float>();
        let v2 = self.mesh.vertices[face[2] as usize].cast::<Float>();
        Triangle {
            vertices: [v0, v1, v2],
        }
        .occlude(ray)
    }
}

pub struct TriangleMeshInstance {
    pub accel: Arc<accel::bvh::BVHAccelerator<TriangleMeshAccelData>>,
    pub bsdf: Arc<dyn Bsdf>,
    pub area: Float,
    pub dist: Distribution1D,
}
impl_base!(TriangleMeshInstance);
pub struct MeshInstanceProxy {
    pub mesh: Arc<TriangleMesh>,
    pub bsdf: Arc<dyn Bsdf>,
}
impl_base!(MeshInstanceProxy);
pub struct AggregateProxy {
    pub shapes: Vec<Arc<dyn Shape>>,
}
impl_base!(AggregateProxy);
impl Shape for AggregateProxy {
    fn intersect<'a>(&'a self, _ray: &Ray) -> Option<Intersection<'a>> {
        panic!("shouldn't be called on cpu")
    }

    fn occlude(&self, _ray: &Ray) -> bool {
        panic!("shouldn't be called on cpu")
    }

    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        panic!("shouldn't be called on cpu")
    }

    fn aabb(&self) -> Bounds3f {
        panic!("shouldn't be called on cpu")
    }

    fn sample_surface(&self, _u: &Vec3) -> SurfaceSample {
        panic!("shouldn't be called on cpu")
    }

    fn area(&self) -> Float {
        panic!("shouldn't be called on cpu")
    }
}
impl Shape for MeshInstanceProxy {
    fn intersect<'a>(&'a self, _ray: &Ray) -> Option<Intersection<'a>> {
        panic!("shouldn't be called on cpu")
    }

    fn occlude(&self, _ray: &Ray) -> bool {
        panic!("shouldn't be called on cpu")
    }

    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        Some(self.bsdf.as_ref())
    }

    fn aabb(&self) -> Bounds3f {
        panic!("shouldn't be called on cpu")
    }

    fn sample_surface(&self, _u: &Vec3) -> SurfaceSample {
        panic!("shouldn't be called on cpu")
    }

    fn area(&self) -> Float {
        self.mesh.area()
    }
}

impl Shape for TriangleMeshInstance {
    fn aabb(&self) -> Bounds3f {
        self.accel.aabb
    }
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>> {
        if let Some(isct) = self.accel.intersect(ray) {
            let idx = isct.prim_id as usize;
            let mesh = &*self.accel.data.mesh;
            let face = mesh.indices[idx as usize];
            let v0 = mesh.vertices[face[0] as usize].cast::<Float>();
            let v1 = mesh.vertices[face[1] as usize].cast::<Float>();
            let v2 = mesh.vertices[face[2] as usize].cast::<Float>();
            let trig = Triangle {
                vertices: [v0, v1, v2],
            };
            let ng = trig.ng();
            let (tc0, tc1, tc2) = self.accel.data.get_tc(&face);

            Some(Intersection {
                shape: Some(self),
                texcoords: lerp3(&tc0, &tc1, &tc2, &isct.uv),
                ng,
                ns: ng,
                ..isct
            })
        } else {
            None
        }
    }
    fn occlude(&self, ray: &Ray) -> bool {
        self.accel.occlude(ray)
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        Some(self.bsdf.as_ref())
    }
    fn area(&self) -> Float {
        self.area
    }
    fn sample_surface(&self, u: &Vec3) -> SurfaceSample {
        self.accel.data.mesh.sample_surface(u, &self.dist)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TriangleMesh {
    pub name: String,
    pub vertices: Vec<glm::Vec3>,
    pub normals: Vec<glm::Vec3>,
    pub texcoords: Vec<glm::Vec2>,
    pub indices: Vec<glm::UVec3>,
}
impl TriangleMesh {
    pub fn sample_surface(&self, u: &Vec3, dist: &Distribution1D) -> SurfaceSample {
        let (idx, pdf_idx) = dist.sample_discrete(u[2]);
        let face = self.indices[idx as usize];
        let v0 = self.vertices[face[0] as usize].cast::<Float>();
        let v1 = self.vertices[face[1] as usize].cast::<Float>();
        let v2 = self.vertices[face[2] as usize].cast::<Float>();
        let trig = Triangle {
            vertices: [v0, v1, v2],
        };
        let uv = uniform_sample_triangle(&vec2(u.x, u.y));
        let p = lerp3(&v0, &v1, &v2, &uv);
        let (tc0, tc1, tc2) = self.get_tc(&face);
        SurfaceSample {
            p,
            ng: trig.ng(),
            ns: trig.ng(),
            texcoords: lerp3(&tc0, &tc1, &tc2, &uv),
            pdf: 1.0 / self.get_triangle(idx).area() * pdf_idx,
        }
    }
    pub fn get_triangle(&self, i: usize) -> Triangle {
        let face = self.indices[i];
        let v0 = self.vertices[face[0] as usize].cast::<Float>();
        let v1 = self.vertices[face[1] as usize].cast::<Float>();
        let v2 = self.vertices[face[2] as usize].cast::<Float>();
        Triangle {
            vertices: [v0, v1, v2],
        }
    }
    pub fn area(&self) -> Float {
        self.indices
            .iter()
            .enumerate()
            .map(|(i, _face)| self.get_triangle(i).area())
            .sum()
    }
    pub fn area_distribution(&self) -> Distribution1D {
        let f: Vec<_> = self
            .indices
            .iter()
            .enumerate()
            .map(|(i, _face)| self.get_triangle(i).area())
            .collect();
        Distribution1D::new(f.as_slice()).unwrap()
    }
    pub fn get_tc(&self, face: &glm::UVec3) -> (Vec2, Vec2, Vec2) {
        if self.texcoords.is_empty() {
            let tc0 = vec2(0.0, 0.0);
            let tc1 = vec2(0.0, 1.0);
            let tc2 = vec2(1.0, 1.0);
            (tc0, tc1, tc2)
        } else {
            let tc0 = self.texcoords[face[0] as usize].cast::<Float>();
            let tc1 = self.texcoords[face[1] as usize].cast::<Float>();
            let tc2 = self.texcoords[face[2] as usize].cast::<Float>();
            (tc0, tc1, tc2)
        }
    }
}

impl TriangleMesh {
    pub fn build_accel(self: &Arc<TriangleMesh>) -> BVHAccelerator<TriangleMeshAccelData> {
        let accel = SweepSAHBuilder::build(
            TriangleMeshAccelData { mesh: self.clone() },
            (0..self.indices.len() as u32).collect(),
        )
        .optimize_layout();
        accel
    }
    pub fn create_instance(
        bsdf: Arc<dyn Bsdf>,
        accel: Arc<BVHAccelerator<TriangleMeshAccelData>>,
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
pub fn compute_normals(model: &mut TriangleMesh) {
    model.normals.clear();
    let mut face_normals = vec![];
    let mut vertex_neighbors: HashMap<u32, Vec<u32>> = HashMap::new();
    for f in 0..model.indices.len() {
        let face = model.indices[f];
        // let face= glm::IVec3::from_rows(&[model.F.row(f)]);//[3 * f..3 * f + 3];
        for idx in face.iter() {
            if !vertex_neighbors.contains_key(idx) {
                vertex_neighbors.insert(*idx, vec![f as u32]);
            } else {
                vertex_neighbors.get_mut(idx).unwrap().push(f as u32);
            }
        }
        let triangle: Vec<glm::Vec3> = face
            .into_iter()
            .map(|idx| model.vertices[*idx as usize])
            .collect();
        let edge0: glm::Vec3 = triangle[1] - triangle[0];
        let edge1: glm::Vec3 = triangle[2] - triangle[0];
        let ng = glm::normalize(&glm::cross(&edge0, &edge1));
        face_normals.push(ng);
    }

    model.normals = (0..model.vertices.len())
        .into_iter()
        .map(|v| match vertex_neighbors.get(&(v as u32)) {
            None => glm::vec3(0.0, 0.0, 0.0),

            Some(faces) => {
                let ng_sum: glm::Vec3 = faces.into_iter().map(|f| face_normals[*f as usize]).sum();
                let ng = ng_sum / (faces.len() as f32);
                ng
            }
        })
        .collect();
}

pub fn load_model(obj_file: &str) -> (Vec<TriangleMesh>, Vec<tobj::Model>, Vec<tobj::Material>) {
    let (models, materials) = tobj::load_obj(&obj_file, true).expect("Failed to load file");

    let mut imported_models = vec![];
    // println!("# of models: {}", models.len());
    // println!("# of materials: {}", materials.len());
    for (_i, m) in models.iter().enumerate() {
        let mesh = &m.mesh;
        if m.name.is_empty() {
            println!(
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
            vertices.push(glm::vec3(
                mesh.positions[3 * v],
                mesh.positions[3 * v + 1],
                mesh.positions[3 * v + 2],
            ));
        }
        let mut indices = vec![];
        for f in 0..mesh.indices.len() / 3 {
            indices.push(uvec3(
                mesh.indices[3 * f],
                mesh.indices[3 * f + 1],
                mesh.indices[3 * f + 2],
            ));
        }
        if !mesh.normals.is_empty() {
            for i in 0..mesh.normals.len() / 3 {
                normals.push(glm::vec3(
                    mesh.normals[3 * i],
                    mesh.normals[3 * i + 1],
                    mesh.normals[3 * i + 2],
                ));
            }
        }
        if !mesh.texcoords.is_empty() {
            for i in 0..mesh.texcoords.len() / 2 {
                texcoords.push(glm::vec2(mesh.texcoords[2 * i], mesh.texcoords[2 * i + 1]));
            }
        }
        let mut imported = TriangleMesh {
            name: m.name.clone(),
            vertices,
            normals,
            indices,
            texcoords,
        };
        if mesh.normals.is_empty() {
            compute_normals(&mut imported);
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
