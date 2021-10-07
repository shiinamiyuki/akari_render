use crate::camera::*;
use crate::light::*;
use crate::shape::*;
use crate::Intersection;
use crate::Ray;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
pub struct Scene {
    pub shape: Arc<dyn Shape>,
    pub camera: Arc<dyn Camera>,
    pub lights: Vec<Arc<dyn Light>>,
    pub light_distr: Arc<dyn LightDistribution>,
    pub ray_counter: AtomicU64,
    pub shape_to_light: HashMap<usize, Arc<dyn Light>>,
    pub meshes: Vec<Arc<TriangleMesh>>,
}

impl Scene {
    pub fn get_light_of_shape<'a>(&'a self, shape: &dyn Shape) -> Option<&'a dyn Light> {
        if let Some(light) = self
            .shape_to_light
            .get(&((shape as *const dyn Shape).cast::<()>() as usize))
        {
            Some(light.as_ref())
        } else {
            None
        }
    }
    pub fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>> {
        self.ray_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.shape.intersect(ray)
    }
    pub fn occlude(&self, ray: &Ray) -> bool {
        self.ray_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.shape.occlude(ray)
    }
}
