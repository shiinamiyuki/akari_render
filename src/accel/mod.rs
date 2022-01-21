use glam::BVec4A;

use crate::bsdf::*;
use crate::shape::*;
use crate::*;
use std::collections::HashSet;

use self::bvh::BvhAccel;
use self::bvh::SweepSAHBuilder;
#[macro_use]
pub mod bvh;
#[cfg(feature = "embree")]
pub mod embree;
pub mod qbvh;
impl_base!(Arc<dyn Shape>);
impl Shape for Arc<dyn Shape> {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<RayHit> {
        self.as_ref().intersect(ray)
    }
    fn occlude(&self, ray: &Ray) -> bool {
        self.as_ref().occlude(ray)
    }
    fn aabb(&self) -> Bounds3f {
        self.as_ref().aabb()
    }
    fn sample_surface(&self, u: Vec3) -> SurfaceSample {
        self.as_ref().sample_surface(u)
    }
    fn area(&self) -> f32 {
        self.as_ref().area()
    }

    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        self.as_ref().bsdf()
    }

    fn shading_triangle<'a>(&'a self, prim_id: u32) -> ShadingTriangle<'a> {
        self.as_ref().shading_triangle(prim_id)
    }

    fn triangle(&self, prim_id: u32) -> Triangle {
        self.as_ref().triangle(prim_id)
    }
}
struct TopLevelBvhData {
    shapes: Vec<Arc<dyn Shape>>,
}

impl bvh::BvhData for TopLevelBvhData {
    fn aabb(&self, idx: u32) -> Bounds3f {
        self.shapes[idx as usize].aabb()
    }
}

pub trait Accel: Send + Sync {
    fn hit_to_iteraction<'a>(&'a self, hit: RayHit) -> SurfaceInteraction<'a>;
    fn intersect(&self, ray: &Ray) -> Option<RayHit>;

    fn occlude(&self, ray: &Ray) -> bool;
    fn shapes(&self) -> Vec<Arc<dyn Shape>>;
    fn intersect4(&self, rays: &[Ray; 4], mask: [bool; 4]) -> [Option<RayHit>; 4] {
        let mut hits = [None; 4];
        for i in 0..4 {
            if mask[i] {
                hits[i] = self.intersect(&rays[i]);
            }
        }
        hits
    }
    fn occlude4(&self, rays: &[Ray; 4], mask: [bool; 4]) -> [bool; 4] {
        let mut occluded = [false; 4];
        for i in 0..4 {
            if mask[i] {
                occluded[i] = self.occlude(&rays[i]);
            }
        }
        occluded
    }
}

pub fn build_accel(shapes: &Vec<Arc<dyn Shape>>, accel: &str) -> Arc<dyn Accel> {
    if accel == "bvh" || accel == "qbvh" {
        build_accel_custom_bvh(shapes, accel)
    } else if accel == "embree" {
        build_accel_embree(shapes)
    } else {
        panic!("unrecognized accel {}", accel)
    }
}
#[cfg(feature = "embree")]
fn build_accel_embree(shapes: &Vec<Arc<dyn Shape>>) -> Arc<dyn Accel> {
    use embree::*;
    Arc::new(unsafe { EmbreeTopLevelAccel::new(shapes) })
}
#[cfg(not(feature = "embree"))]
fn build_accel_embree(shapes: &Vec<Arc<dyn Shape>>) -> Arc<dyn Accel> {
    unimplemented!()
}
fn build_accel_custom_bvh(shapes: &Vec<Arc<dyn Shape>>, accel_type: &str) -> Arc<dyn Accel> {
    let mut cache: HashMap<*const dyn Any, Arc<MeshBvh>> = HashMap::new();
    let shapes: Vec<_> = shapes
        .iter()
        .map(|shape_| {
            let shape = shape_.as_ref().as_any();
            if let Some(mesh) = shape.downcast_ref::<MeshInstanceProxy>() {
                let base = mesh.mesh.clone();
                if !cache.contains_key(&Arc::as_ptr(&(base.clone() as Arc<dyn Any>))) {
                    let accel = base.clone().build_accel();
                    let accel = match accel_type {
                        "bvh" => MeshBvh::Bvh(accel),
                        "qbvh" => MeshBvh::QBvh(qbvh::QBvhAccelBuilder::new(accel).build()),
                        _ => unreachable!(),
                    };
                    cache.insert(
                        Arc::as_ptr(&(base.clone() as Arc<dyn Any>)),
                        Arc::new(accel),
                    );
                }
                let accel = cache
                    .get(&Arc::as_ptr(&(base.clone() as Arc<dyn Any>)))
                    .unwrap()
                    .clone();
                TriangleMesh::create_instance(mesh.bsdf.clone(), accel, base.clone())
            } else {
                shape_.clone()
            }
        })
        .collect();
    let bvh_data = TopLevelBvhData {
        shapes: shapes.clone(),
    };
    let bvh: BvhAccel<TopLevelBvhData> =
        SweepSAHBuilder::build(bvh_data, (0..shapes.len() as u32).collect());
    match accel_type {
        "bvh" => Arc::new(bvh),
        "qbvh" => Arc::new(qbvh::QBvhAccelBuilder::new(bvh).build()),
        _ => unreachable!(),
    }
}
