use crate::{
    bsdf::{BsdfClosure, BsdfFlags},
    film::Film,
    light::{Light, ReferencePoint},
    scene::Scene,
    shape::{Shape, SurfaceInteraction},
    util::profile::scope,
    *,
};
use indicatif::ProgressBar;
use rand::{thread_rng, Rng};
use sampler::{Sampler, SobolSampler};

use super::Integrator;

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
    l: SampledSpectrum,
    beta: SampledSpectrum,
    prev_n: Vec3,
    prev_bsdf_pdf: f32,
    pixel: u32,
    is_delta: bool,
    depth: u32,
}
#[derive(Clone, Copy)]
struct ShadowRay {
    ray: Ray,
    state_idx: u32,
    ld: SampledSpectrum,
}
impl Default for ShadowRay {
    fn default() -> Self {
        Self {
            ray: Default::default(),
            state_idx: u32::MAX,
            ld: SampledSpectrum::zero(),
        }
    }
}
impl ShadowRay {
    fn is_invalid(&self) -> bool {
        self.state_idx == u32::MAX || self.ray.is_invalid()
    }
}

#[derive(Clone, Copy)]
struct ClosestHit {
    ray: Ray,
    hit: RayHit,
    state_idx: u32,
}
impl Default for ClosestHit {
    fn default() -> Self {
        Self {
            ray: Default::default(),
            state_idx: u32::MAX,
            hit: Default::default(),
        }
    }
}
impl ClosestHit {
    fn is_invalid(&self) -> bool {
        self.state_idx == u32::MAX || self.ray.is_invalid()
    }
}
#[derive(Clone, Copy)]
struct BsdfSampleContext<'a> {
    bsdf: BsdfClosure<'a>,
    wo: Vec3,
    si: SurfaceInteraction<'a>,
    p: Vec3,
}
struct StreamPathTracerSession<'a> {
    spp: u32,
    max_depth: u32,
    batch_size: usize,
    sort_rays: bool,
    npixels: usize,
    ray_id: usize,
    seeds: Vec<u64>,
    scene: &'a Scene,
    film: &'a mut Film,
    progress: &'a ProgressBar,
}
fn mis_weight(mut pdf_a: f32, mut pdf_b: f32) -> f32 {
    pdf_a *= pdf_a;
    pdf_b *= pdf_b;
    pdf_a / (pdf_a + pdf_b)
}
impl<'a> StreamPathTracerSession<'a> {
    #[allow(dead_code)]
    fn sort_rays(&self, path_states: &mut [PathState], rayhits: &mut [ClosestHit]) {
        let _profiler = scope("StreamPathTracerSession::sort_rays");
        rayhits.par_sort_by_key(|rayhit| {
            let mut k = 0;
            for i in 0..3 {
                k |= (if rayhit.ray.d[i] < 0.0 { 0 } else { 1 }) << i;
            }
            k
        });
        util::par_permute(path_states, |i| rayhits[i].state_idx as usize);
    }
    #[allow(dead_code)]
    fn sort_shadow_rays(&self, shadow_rays: &mut [ShadowRay]) {
        let _profiler = scope("StreamPathTracerSession::sort_shadow_rays");
        shadow_rays.par_sort_by_key(|shadow_ray| {
            let mut k = 0;
            for i in 0..3 {
                k |= (if shadow_ray.ray.d[i] < 0.0 { 0 } else { 1 }) << i;
            }
            k
        });
    }
    fn render(&mut self) {
        let mut path_states = vec![];
        let mut rayhits = vec![];
        let mut shadow_rays = vec![];
        loop {
            let new_states = self.batch_size - path_states.len();
            if new_states > 0 {
                self.generate_rays(new_states, &mut path_states, &mut rayhits);
            }
            if path_states.is_empty() {
                break;
            }
            self.intersect(&mut rayhits);
            self.eval_materials(&mut path_states, &mut rayhits, &mut shadow_rays);
            self.trace_shadow_rays(&mut path_states, &mut shadow_rays);
        }
    }
    fn intersect(&self, items: &mut [ClosestHit]) {
        let _profiler = scope("StreamPathTracerSession::intersect");
        parallel_for_slice_packet(items, 1024, 4, |_, item| {
            let accel = &self.scene.accel;
            let mut ray4 = [Ray::default(); 4];
            let mut mask = [false; 4];
            for i in 0..item.len() {
                ray4[i] = item[i].ray;
                mask[i] = !item[i].is_invalid();
            }
            let hits = accel.intersect4(&ray4, mask);
            for i in 0..item.len() {
                item[i].hit = hits[i].unwrap_or(Default::default());
            }
            self.scene
                .ray_counter
                .fetch_add(item.len() as u64, Ordering::Relaxed);
        });
    }
    fn trace_shadow_rays(&self, path_states: &mut [PathState], shadow_rays: &mut [ShadowRay]) {
        let _profiler = scope("StreamPathTracerSession::trace_shadow_rays");
        let p_path_states = UnsafePointer::new(path_states.as_mut_ptr());
        parallel_for_slice_packet(shadow_rays, 1024, 4, |p, shadow_ray| {
            let accel = &self.scene.accel;
            let mut ray4 = [Ray::default(); 4];
            let mut mask = [false; 4];
            for i in 0..shadow_ray.len() {
                ray4[i] = shadow_ray[i].ray;
                mask[i] = !shadow_ray[i].is_invalid();
            }
            let occluded = accel.occlude4(&ray4, mask);
            for i in 0..shadow_ray.len() {
                if mask[i] {
                    let path_state = unsafe { &mut *p_path_states.p.offset((p * 4 + i) as isize) };
                    if !occluded[i] {
                        path_state.l += shadow_ray[i].ld;
                    }
                }
            }
            self.scene
                .ray_counter
                .fetch_add(shadow_ray.len() as u64, Ordering::Relaxed);
        });
    }
    fn eval_materials(
        &self,
        path_states: &mut Vec<PathState>,
        rayhits: &mut Vec<ClosestHit>,
        shadow_rays: &mut Vec<ShadowRay>,
    ) {
        let _profiler = scope("StreamPathTracerSession::eval_materials");
        let mut bsdfs: Vec<Option<BsdfSampleContext<'a>>> = vec![None; path_states.len()];
        parallel_for_slice3(
            path_states,
            rayhits,
            &mut bsdfs,
            1024,
            |i, path_state, rayhits, bsdf_ctx| {
                if rayhits.is_invalid() {
                    return;
                }
                assert_eq!(i, rayhits.state_idx as usize);
                let prev_ray = rayhits.ray;
                let ray = prev_ray;
                rayhits.ray = Ray::default();
                let hit = rayhits.hit;
                if hit.is_invalid() {
                    return;
                }
                rayhits.hit = Default::default();
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
                let depth = path_state.depth;
                if let Some(light) = self.scene.get_light_of_shape(shape) {
                    if depth == 0 {
                        path_state.l += path_state.beta * light.le(&ray);
                    } else {
                        if depth > 1 {
                            let light_pdf = self.scene.light_distr.pdf(light)
                                * light
                                    .pdf_li(
                                        ray.d,
                                        &ReferencePoint {
                                            p: ray.o,
                                            n: path_state.prev_n,
                                        },
                                    )
                                    .1;
                            let bsdf_pdf = path_state.prev_bsdf_pdf;
                            assert!(light_pdf.is_finite());
                            assert!(light_pdf >= 0.0);
                            let weight = if path_state.is_delta {
                                1.0
                            } else {
                                mis_weight(bsdf_pdf, light_pdf)
                            };

                            path_state.l += path_state.beta * light.le(&ray) * weight;
                        }
                    }
                }
                let wo = -ray.d;

                if path_state.depth >= self.max_depth {
                    return;
                }
                path_state.depth += 1;
                *bsdf_ctx = Some(BsdfSampleContext { p, wo, bsdf, si });
            },
        );
        {
            let mut i = 0;
            let mut done = 0;
            while i < path_states.len() {
                if bsdfs[i].is_none() {
                    let state = path_states[i];
                    {
                        let pixel_id = state.pixel;
                        let py = pixel_id / self.scene.camera.resolution().x;
                        let px = pixel_id % self.scene.camera.resolution().x;
                        self.film.add_sample(uvec2(px, py), &state.l, 1.0);
                    }
                    let last = path_states.len() - 1;
                    bsdfs.swap(i, last);
                    bsdfs.pop();
                    path_states.swap(i, last);
                    path_states.pop();
                    done += 1;
                } else {
                    i += 1;
                }
            }
            self.progress.inc(done);
            rayhits.resize(path_states.len(), ClosestHit::default());
            shadow_rays.resize(path_states.len(), ShadowRay::default());
        }
        parallel_for_slice4(
            path_states,
            &mut bsdfs,
            shadow_rays,
            rayhits,
            1024,
            |i, path_state, bsdf_ctx, shadow_ray, rayhit| {
                *shadow_ray = Default::default();
                let BsdfSampleContext { p, wo, bsdf, si } = bsdf_ctx.unwrap();
                let shape = si.shape;
                let scene = self.scene;
                let sampler = &mut path_state.sampler;
                {
                    let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
                    let sample_self = if let Some(light2) = scene.get_light_of_shape(shape) {
                        if light as *const dyn Light == light2 as *const dyn Light {
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if !sample_self {
                        let p_ref = ReferencePoint { p, n: si.ng };
                        let light_sample = light.sample_li(sampler.next3d(), &p_ref);
                        let light_pdf = light_sample.pdf * light_pdf;
                        let bsdf_pdf = bsdf.evaluate_pdf(wo, light_sample.wi);
                        let weight = if light.is_delta() {
                            1.0
                        } else {
                            mis_weight(light_pdf, bsdf_pdf)
                        };
                        let ld = path_state.beta
                            * bsdf.evaluate(wo, light_sample.wi)
                            * si.ng.dot(light_sample.wi).abs()
                            * light_sample.li
                            / light_pdf
                            * weight;
                        *shadow_ray = ShadowRay {
                            state_idx: i as u32,
                            ray: light_sample.shadow_ray,
                            ld,
                        };
                    }
                }
                if let Some(bsdf_sample) = bsdf.sample(sampler.next2d(), wo) {
                    path_state.is_delta = bsdf_sample.flag.contains(BsdfFlags::SPECULAR);
                    let wi = bsdf_sample.wi;
                    let ray = Ray::spawn(p, wi).offset_along_normal(si.ng);
                    path_state.beta *= bsdf_sample.f * wi.dot(si.ng).abs() / bsdf_sample.pdf;
                    path_state.prev_bsdf_pdf = bsdf_sample.pdf;
                    path_state.prev_n = si.ng;
                    *rayhit = ClosestHit {
                        ray,
                        hit: Default::default(),
                        state_idx: i as u32,
                    };
                } else {
                    *rayhit = Default::default();
                }
            },
        );
    }
    fn generate_rays(
        &mut self,
        count: usize,
        path_states: &mut Vec<PathState>,
        rayhits: &mut Vec<ClosestHit>,
    ) {
        let ray_id = self.ray_id;
        let total = self.npixels * self.spp as usize;
        self.ray_id = (self.ray_id + count).min(total);
        let count = self.ray_id - ray_id;
        let new_len = path_states.len() + count;
        let old_len = path_states.len();
        assert_eq!(path_states.len(), rayhits.len());
        let new_states: Vec<_> = (ray_id..self.ray_id)
            .into_par_iter()
            .map(|id| {
                let pixel_id = id / self.spp as usize;
                let mut sampler = SobolSampler::new(self.seeds[pixel_id]);
                sampler.index = (id % self.spp as usize) as u32;
                PathState {
                    sampler,
                    depth: 0,
                    l: SampledSpectrum::zero(),
                    beta: SampledSpectrum::one(),
                    prev_n: Vec3::ZERO,
                    prev_bsdf_pdf: 0.0,
                    is_delta: false,
                    pixel: pixel_id as u32,
                }
            })
            .collect();
        path_states.extend_from_slice(&new_states);
        std::mem::drop(new_states);
        rayhits.resize(new_len, Default::default());
        parallel_for_slice2(
            &mut path_states[old_len..new_len],
            &mut rayhits[old_len..new_len],
            1024,
            |i, state, rayhit| {
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
                *rayhit = ClosestHit {
                    ray,
                    state_idx: (i + old_len) as u32,
                    hit: RayHit::default(),
                };
            },
        )
    }
}

impl Integrator for StreamPathTracer {
    fn render(&self, scene: &Scene) -> film::Film {
        log::info!("rendering {}spp ... with StreamPathTracer", self.spp);
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let mut film = Film::new(&scene.camera.resolution());
        let total = npixels * self.spp as usize;
        let progress = crate::util::create_progess_bar(total, "samples");
        {
            let mut session = StreamPathTracerSession {
                spp: self.spp,
                film: &mut film,
                progress: &progress,
                max_depth: self.max_depth,
                batch_size: self.batch_size,
                sort_rays: self.sort_rays,
                npixels,
                ray_id: 0,
                seeds: (0..npixels)
                    .map(|_| {
                        let mut rng = thread_rng();
                        rng.gen::<u64>()
                    })
                    .collect(),
                scene,
            };
            session.render();
        }
        film
    }
}
