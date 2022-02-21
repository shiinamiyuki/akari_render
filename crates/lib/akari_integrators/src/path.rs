use crate::bsdf::*;
use crate::film::*;
use crate::light::*;
use crate::sampler::*;
use crate::scene::*;
use crate::util::profile::scope;
use crate::util::PerThread;
use crate::*;
use bumpalo::Bump;

pub struct PathTracer {
    pub spp: u32,
    pub max_depth: u32,
    pub single_wavelength: bool,
}
fn mis_weight(mut pdf_a: f32, mut pdf_b: f32) -> f32 {
    pdf_a *= pdf_a;
    pdf_b *= pdf_b;
    pdf_a / (pdf_a + pdf_b)
}
impl PathTracer {
    pub fn li(
        mut ray: Ray,
        lambda: &mut SampledWavelengths,
        sampler: &mut dyn Sampler,
        scene: &Scene,
        max_depth: usize,
        indirect_only: bool,
        arena: &Bump,
    ) -> SampledSpectrum {
        let mut li = SampledSpectrum::zero();
        let mut beta = SampledSpectrum::one();
        let mut prev_n: Option<Vec3> = None;
        let mut prev_bsdf_pdf: Option<f32> = None;
        let mut is_delta = false;
        {
            let mut depth = 0;
            loop {
                if let Some(si) = scene.intersect(&ray) {
                    let ng = si.ng;
                    let ns = si.ns;
                    let shape = si.shape;
                    let opt_bsdf = si.evaluate_bsdf(lambda, TransportMode::CameraToLight, arena);
                    if opt_bsdf.is_none() {
                        break;
                    }
                    let p = ray.at(si.t);
                    let bsdf = opt_bsdf.unwrap();
                    let _profiler = scope("PathTracer::li::<env hit>");
                    if let Some(light) = scene.get_light_of_shape(shape) {
                        // li += beta * light.le(&ray);
                        if depth == 0 {
                            if !indirect_only {
                                li += beta * light.emission(&ray, lambda);
                            }
                        } else {
                            if !indirect_only || depth > 1 {
                                let light_pdf = scene.light_distr.pdf(light)
                                    * light
                                        .pdf_direct(
                                            ray.d,
                                            &ReferencePoint {
                                                p: ray.o,
                                                n: prev_n.unwrap(),
                                            },
                                        )
                                        .1;
                                let bsdf_pdf = prev_bsdf_pdf.unwrap();
                                assert!(light_pdf.is_finite());
                                assert!(light_pdf >= 0.0);
                                let weight = if is_delta {
                                    1.0
                                } else {
                                    mis_weight(bsdf_pdf, light_pdf)
                                };

                                li += beta * light.emission(&ray, lambda) * weight;
                            }
                        }
                    }
                    std::mem::drop(_profiler);

                    let wo = -ray.d;

                    if depth >= max_depth {
                        break;
                    }
                    depth += 1;
                    {
                        let _profiler = scope("PathTracer::li::<light sampling>");
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
                            let p_ref = ReferencePoint { p, n: ng };
                            let light_sample = light.sample_direct(sampler.next3d(), &p_ref, lambda);
                            let light_pdf = light_sample.pdf * light_pdf;
                            if (!indirect_only || depth > 1)
                                && light_pdf > 0.0
                                && light_pdf.is_finite()
                            {
                                if !light_sample.li.is_black()
                                    && !scene.occlude(&light_sample.shadow_ray)
                                {
                                    let bsdf_pdf = bsdf.evaluate_pdf(wo, light_sample.wi);
                                    let weight = if light.is_delta() {
                                        1.0
                                    } else {
                                        mis_weight(light_pdf, bsdf_pdf)
                                    };
                                    // println!("{} {} {}", light_pdf, bsdf_pdf, weight);
                                    // assert!(light_pdf.is_finite());
                                    assert!(light_pdf >= 0.0);
                                    li += beta
                                        * bsdf.evaluate(wo, light_sample.wi)
                                        * ns.dot(light_sample.wi).abs()
                                        * light_sample.li
                                        / light_pdf
                                        * weight;
                                }
                            }
                        }
                    }

                    {
                        let _profiler = scope("PathTracer::li::<bsdf sampling>");
                        if let Some(bsdf_sample) = bsdf.sample(sampler.next2d(), wo) {
                            is_delta = bsdf_sample.flag.contains(BsdfFlags::SPECULAR);
                            let wi = bsdf_sample.wi;
                            ray = Ray::spawn(p, wi).offset_along_normal(ng);
                            beta *= bsdf_sample.f * wi.dot(ns).abs() / bsdf_sample.pdf;
                            prev_bsdf_pdf = Some(bsdf_sample.pdf);
                            prev_n = Some(si.ng);
                        } else {
                            break;
                        }
                    }
                } else {
                    break;
                }
            }
        }
        if li.is_black() {
            SampledSpectrum::zero()
        } else {
            li
        }
    }
}
impl Integrator for PathTracer {
    fn render(&self, scene: &Scene) -> Film {
        log::info!("rendering {}spp ...", self.spp);
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        let chunks = (npixels + 255) / 256;
        let progress = crate::util::create_progess_bar(chunks, "chunks");
        let arenas = PerThread::new(|| Bump::new());

        parallel_for(npixels, 256, |id| {
            let mut sampler = SobolSampler::new(id as u64);
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let arena = arenas.get_mut();
            for _ in 0..self.spp {
                {
                    let arena = &*arena;
                    sampler.start_next_sample();
                    let mut lambda = SampledWavelengths::sample_visible(sampler.next1d());
                    if self.single_wavelength {
                        lambda.terminate_secondary();
                    }
                    let (ray, _ray_weight) =
                        scene.camera.generate_ray(pixel, &mut sampler, &lambda);
                    let li = Self::li(
                        ray,
                        &mut lambda,
                        &mut sampler,
                        scene,
                        self.max_depth as usize,
                        false,
                        arena,
                    );
                    film.add_sample(uvec2(x, y), li, lambda, 1.0);
                }
                arena.reset();
            }

            if (id + 1) % 256 == 0 {
                progress.inc(1);
            }
        });
        progress.finish();
        film
    }
}
