use crate::accel;
use crate::accel::Accel;
use crate::camera::*;
use crate::light::*;
use crate::shape::*;
use crate::Ray;
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
pub struct Scene {
    pub accel: Arc<dyn Accel>,
    pub camera: Arc<dyn Camera>,
    pub lights: Vec<Arc<dyn Light>>,
    pub light_distr: Arc<dyn LightDistribution>,
    pub ray_counter: AtomicU64,
    pub shape_to_light: HashMap<usize, Arc<dyn Light>>,
    pub meshes: Vec<Arc<TriangleMesh>>,
}

impl Scene {
    pub fn new(
        camera: Arc<dyn Camera>,
        shapes: Vec<Arc<dyn Shape>>,
        meshes: Vec<Arc<TriangleMesh>>,
        mut lights: Vec<Arc<dyn Light>>,
        accel: &str,
        is_gpu: bool,
    ) -> Self {
        let toplevel = if is_gpu {
            // Arc::new(AggregateProxy { shapes })
            todo!()
        } else {
            accel::build_accel(&shapes, accel)
        };
        let mut shape_to_light = HashMap::new();
        for shape in toplevel.shapes() {
            if let Some(bsdf) = shape.bsdf() {
                if let Some(emission) = bsdf.emission() {
                    if emission.power() > 0.001 {
                        let light: Arc<dyn Light> = Arc::new(AreaLight {
                            emission,
                            shape: shape.clone(),
                        });
                        lights.push(light.clone());
                        shape_to_light
                            .insert(Arc::into_raw(shape.clone()).cast::<()>() as usize, light);
                    }
                }
            }
        }

        Self {
            ray_counter: AtomicU64::new(0),
            camera,
            lights: lights.clone(),
            shape_to_light,
            light_distr: Arc::new(PowerLightDistribution::new(lights)),
            accel: toplevel,
            meshes,
        }
    }
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
    pub fn intersect<'a>(&'a self, ray: &Ray) -> Option<SurfaceInteraction<'a>> {
        self.ray_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.accel
            .intersect(ray)
            .map(|hit| self.accel.hit_to_iteraction(hit))
    }
    pub fn occlude(&self, ray: &Ray) -> bool {
        self.ray_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.accel.occlude(ray)
    }
}
