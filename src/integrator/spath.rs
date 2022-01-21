use crate::{bsdf::BsdfClosure, scene::Scene, *};
use rand::{thread_rng, Rng};
use sampler::{Sampler, SobolSampler};

// Streaming Path Tracer

pub struct StreamPathTracer {
    pub spp: u32,
    pub max_depth: u32,
    pub batch_size: usize,
    pub sort_rays: bool,
}

#[derive(Clone, Copy)]
struct PathState {
    sampler: SobolSampler,
    l: Spectrum,
    beta: Spectrum,
    prev_n: Vec3,
    prev_bsdf_pdf: f32,
    is_delta: bool,
}
#[derive(Clone, Copy)]
struct ShadowRay {
    ray: Ray,
    state_id: u32,
    occluded: bool,
}

#[derive(Clone, Copy)]
struct ClosestHit {
    ray: Ray,
    hit: RayHit,
    state_id: u32,
}

struct StreamPathTracerSession<'a> {
    spp: u32,
    max_depth: u32,
    batch_size: usize,
    sort_rays: bool,
    npixels: usize,
    ray_id: usize,
    scene: &'a Scene,
}
fn mis_weight(mut pdf_a: f32, mut pdf_b: f32) -> f32 {
    pdf_a *= pdf_a;
    pdf_b *= pdf_b;
    pdf_a / (pdf_a + pdf_b)
}
impl<'a> StreamPathTracerSession<'a> {
    fn intersect(&self, items: &mut [ClosestHit]) {
        parallel_for_slice(items, 1024, |_, item| {
            if item.ray.is_invalid() {
                return;
            }
            item.hit = self
                .scene
                .accel
                .intersect(&item.ray)
                .unwrap_or(Default::default());
        });
        parallel_for_slice(items, 1024, |_, item| {
            
        });
    }
    fn test_shadow_rays(&self, items: &mut [ShadowRay]) {
        parallel_for_slice(items, 1024, |i, item| {
            if item.ray.is_invalid() {
                return;
            }
            item.occluded = self.scene.accel.occlude(&item.ray);
        });
    }
    fn eval_materials(
        &self,
        path_states: &mut [PathState],
        hits: &[RayHit],
        rays: &mut [Ray],
        shadow_rays: &mut [Ray],
    ) {
        parallel_for_slice3(
            path_states,
            rays,
            shadow_rays,
            1024,
            |i, path_state, ray, shadow_ray| {
                let prev_ray = *ray;
                *ray = Ray::default();
                let hit = hits[i];
                if hit.is_invalid() {
                    return;
                }
                let si = self.scene.accel.hit_to_iteraction(hit);
                let ng = si.ng;
                let frame = Frame::from_normal(ng);
                let shape = si.shape;
                let opt_bsdf = si.bsdf;
                if opt_bsdf.is_none() {
                    return;
                }
                let p = ray.at(si.t);
                let bsdf = BsdfClosure {
                    sp: si.sp,
                    frame,
                    bsdf: opt_bsdf.unwrap(),
                };
            },
        );
    }
    fn generate_rays(&mut self) -> (Vec<PathState>, Vec<Ray>) {
        let ray_id = self.ray_id;
        let total = self.npixels * self.spp as usize;
        self.ray_id = self.ray_id.min(self.batch_size).min(total);
        let mut path_states: Vec<_> = (ray_id..self.ray_id)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();

                let sampler = SobolSampler::new(rng.gen());

                PathState {
                    sampler,
                    // ray,
                    l: Spectrum::zero(),
                    beta: Spectrum::one(),
                    prev_n: Vec3::ZERO,
                    prev_bsdf_pdf: 0.0,
                    is_delta: false,
                }
            })
            .collect();
        let rays: Vec<_> = path_states
            .par_iter_mut()
            .enumerate()
            .map(|(i, state)| {
                let id = i + ray_id;
                let pixel_id = id / self.spp as usize;
                let py = pixel_id / self.scene.camera.resolution().x as usize;
                let px = pixel_id % self.scene.camera.resolution().x as usize;
                let sampler = &mut state.sampler;
                sampler.start_next_sample();
                let (ray, _ray_weight) = self
                    .scene
                    .camera
                    .generate_ray(uvec2(px as u32, py as u32), sampler);
                ray
            })
            .collect();
        (path_states, rays)
    }
}
