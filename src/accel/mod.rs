use crate::bsdf::*;
use crate::shape::*;
use crate::*;
use std::collections::HashSet;

use self::bvh::BvhAccelerator;
use self::bvh::SweepSAHBuilder;
pub mod bvh;
#[cfg(feature = "embree")]
pub mod embree;
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
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<SurfaceInteraction<'a>>;
    fn occlude(&self, ray: &Ray) -> bool;
    fn shapes(&self) -> Vec<Arc<dyn Shape>>;
}

pub fn build_accel(shapes: &Vec<Arc<dyn Shape>>, accel: &str) -> Arc<dyn Accel> {
    if accel == "bvh" {
        build_accel_custom_bvh(shapes)
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
fn build_accel_custom_bvh(shapes: &Vec<Arc<dyn Shape>>) -> Arc<dyn Accel> {
    let mut cache: HashMap<*const dyn Any, Arc<BvhAccelerator<TriangleMeshAccelData>>> =
        HashMap::new();
    let shapes: Vec<_> = shapes
        .iter()
        .map(|shape_| {
            let shape = shape_.as_ref().as_any();
            if let Some(mesh) = shape.downcast_ref::<MeshInstanceProxy>() {
                let base = mesh.mesh.clone();
                if !cache.contains_key(&Arc::as_ptr(&(base.clone() as Arc<dyn Any>))) {
                    let accel = base.clone().build_accel();
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
    let bvh: BvhAccelerator<TopLevelBvhData> =
        SweepSAHBuilder::build(bvh_data, (0..shapes.len() as u32).collect());
    Arc::new(bvh)
}
