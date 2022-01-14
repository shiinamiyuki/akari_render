use std::collections::HashSet;
use crate::bsdf::*;
use crate::shape::*;
use crate::*;

use self::bvh::BVHAccelerator;
pub mod bvh;
#[cfg(feature = "embree")]
pub mod embree;
impl_base!(Arc<dyn Shape>);
impl Shape for Arc<dyn Shape> {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>> {
        self.as_ref().intersect(ray)
    }
    fn occlude(&self, ray: &Ray) -> bool {
        self.as_ref().occlude(ray)
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        self.as_ref().bsdf()
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
    fn children(&self) -> Option<Vec<Arc<dyn Shape>>> {
        self.as_ref().children()
    }
}
struct GenericBVHData {
    shapes: Vec<Arc<dyn Shape>>,
}

impl bvh::BVHData for GenericBVHData {
    fn intersect<'a>(&'a self, idx: u32, ray: &Ray) -> Option<Intersection<'a>> {
        self.shapes[idx as usize].intersect(ray)
    }
    fn occlude(&self, idx: u32, ray: &Ray) -> bool {
        self.shapes[idx as usize].occlude(ray)
    }
    fn bsdf<'a>(&'a self, idx: u32) -> Option<&'a dyn Bsdf> {
        self.shapes[idx as usize].bsdf()
    }
    fn aabb(&self, idx: u32) -> Bounds3f {
        self.shapes[idx as usize].aabb()
    }
}

pub struct Aggregate {
    bvh: bvh::BVHAccelerator<GenericBVHData>,
    area: f32,
}
impl Aggregate {
    pub fn shapes(&self) -> impl Iterator<Item = &Arc<dyn Shape>> {
        self.bvh.data.shapes.iter()
    }
    pub fn new(shapes: Vec<Arc<dyn Shape>>) -> Self {
        let v: Vec<u32> = (0..shapes.len() as u32).collect();
        let area: f32 = shapes.iter().map(|s| s.area()).sum();
        let data = GenericBVHData { shapes };
        Self {
            bvh: bvh::SweepSAHBuilder::build(data, v).optimize_layout(),
            area,
        }
    }
}
impl_base!(Aggregate);
impl Shape for Aggregate {
    fn intersect<'a>(&'a self, original_ray: &Ray) -> Option<Intersection<'a>> {
        self.bvh.intersect(original_ray)
    }
    fn occlude(&self, ray: &Ray) -> bool {
        self.bvh.occlude(ray)
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn Bsdf> {
        None
    }
    fn aabb(&self) -> Bounds3f {
        Bounds3f::default()
    }
    fn area(&self) -> f32 {
        self.area
    }
    fn sample_surface(&self, u: Vec3) -> SurfaceSample {
        let len = self.bvh.data.shapes.len() as f32;
        let i = u[2] as f32 * len;
        let i = i as usize;
        let u = vec3(u[0], u[1], (u[2] - i as f32 / len) * (len - i as f32));
        self.bvh.data.shapes[i].sample_surface(u)
    }
    fn children(&self) -> Option<Vec<Arc<dyn Shape>>> {
        Some(self.shapes().map(|x| x.clone()).collect())
    }
}

pub fn build_accel(shape: Arc<dyn Shape>, accel: &str) -> Arc<dyn Shape> {
    if accel == "bvh" {
        build_accel_custom_bvh(shape)
    } else if accel == "embree" {
        build_accel_embree(shape)
    } else {
        panic!("unrecognized accel {}", accel)
    }
}
#[cfg(feature = "embree")]
fn build_accel_embree(shape: Arc<dyn Shape>) -> Arc<dyn Shape> {
    use embree::*;
    Arc::new(unsafe { EmbreeTopLevelAccel::new(shape) })
}
#[cfg(not(feature = "embree"))]
fn build_accel_embree(shape: Arc<dyn Shape>) -> Arc<dyn Shape> {
    unimplemented!()
}
fn build_accel_custom_bvh(shape: Arc<dyn Shape>) -> Arc<dyn Shape> {
    let aggregate = shape
        .as_ref()
        .as_any()
        .downcast_ref::<AggregateProxy>()
        .unwrap_or_else(|| {
            panic!("expected AggregateProxy but found {:?}", shape.type_name());
        });
    let mut cache: HashMap<*const dyn Any, Arc<BVHAccelerator<TriangleMeshAccelData>>> =
        HashMap::new();
    let shapes: Vec<_> = aggregate
        .shapes
        .iter()
        .map(|shape_| {
            let shape = shape_.as_ref().as_any();
            assert!(!shape.is::<AggregateProxy>());
            assert!(!shape.is::<Aggregate>());
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
    Arc::new(Aggregate::new(shapes))
}
