use crate::path::PathTracer;
use crate::pssmlt::Pssmlt;
use crate::sampler::{MltSampler, PrimarySample, ReplaySampler, Sampler, SobolSampler};
use crate::scene::Scene;
use crate::util::PerThread;
use crate::*;
use bumpalo::Bump;
use pssmlt::Chain;
use rand::{thread_rng, Rng};

pub struct Erpt {
    pub spp: u32,
    pub direct_spp: u32,
    pub max_depth: u32,
    pub n_bootstrap: usize,
    pub mutations_per_chain: usize,
    // pub n_chains: usize,
}

fn target_function(s: SampledSpectrum, lambda: &SampledWavelengths) -> f32 {
    lambda.cie_xyz(s).values().y.clamp(0.0, 100.0)
}
impl Integrator for Erpt {
    fn render(&self, scene: &Scene) -> Film {
        log::info!("rendering direct lighting...");
        let mut depth0_pt = PathTracer {
            spp: self.direct_spp,
            max_depth: 1,
            single_wavelength:false,
        };
        let film_direct = depth0_pt.render(scene);
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        log::info!("bootstrapping...");
        let mut arena = Bump::new();
        let (_, b) = Pssmlt::init_chain(
            self.max_depth as usize,
            self.n_bootstrap,
            0,
            scene,
            &mut arena,
        );
        let e_avg = b / self.n_bootstrap as f32;
        log::info!("average energy: {}", e_avg);
        let e_d = e_avg / self.mutations_per_chain as f32;
        log::info!("deposit energy: {}", e_d);
        let indirect_film = Film::new(&scene.camera.resolution());
        let chunks = (npixels + 255) / 256;
        let progress = crate::util::create_progess_bar(chunks, "chunks");
        let arenas = PerThread::new(|| Bump::new());
        parallel_for(npixels, 256, |id| {
            let mut sampler = ReplaySampler::new(SobolSampler::new(id as u64));
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let mut acc_li = SampledSpectrum::zero();
            let mut rng = thread_rng();
            let arena = arenas.get_mut();
            for _ in 0..self.spp {
                sampler.start_next_sample();
                let mut lambda = SampledWavelengths::sample_visible(sampler.next1d());
                let (ray, _ray_weight) = scene.camera.generate_ray(pixel, &mut sampler, &lambda);
                let li = PathTracer::li(
                    ray,
                    &mut lambda,
                    &mut sampler,
                    scene,
                    self.max_depth as usize,
                    true,
                    arena,
                );
                let e = target_function(li, &lambda);
                let mean_chains = e / (self.mutations_per_chain as f32 * e_d);
                let dep_energy =
                    e / (self.spp as f32 * mean_chains * self.mutations_per_chain as f32);
                {
                    let num_chains = (rng.gen::<f32>() + mean_chains).floor() as usize;
                    for _ in 0..num_chains {
                        let mut mlt_sampler = MltSampler::from_replay(&sampler, rng.gen());
                        {
                            let px = (x as f32 + 0.5) / scene.camera.resolution().x as f32;
                            let py = (y as f32 + 0.5) / scene.camera.resolution().y as f32;
                            mlt_sampler.samples.insert(0, PrimarySample::new(px));
                            mlt_sampler.samples.insert(1, PrimarySample::new(py));
                        }
                        let mut chain = Chain {
                            sampler: mlt_sampler,
                            large_step_prob: 0.3,
                            is_large_step: false,
                            cur: mmlt::FRecord {
                                pixel,
                                f: e,
                                l: li,
                                lambda: lambda.clone(),
                            },
                            max_depth: self.max_depth as usize,
                        };
                        for _ in 0..self.mutations_per_chain {
                            let proposal = chain.run(scene, arena);
                            let accept_prob = match chain.cur.f {
                                x if x > 0.0 => (proposal.f / x).min(1.0),
                                _ => 1.0,
                            };
                            if proposal.f > 0.0 {
                                let dep_value =
                                    (proposal.l / proposal.f) * dep_energy * accept_prob;
                                indirect_film.add_sample(
                                    proposal.pixel,
                                    dep_value,
                                    proposal.lambda.clone(),
                                    1.0,
                                );
                            }
                            if chain.cur.f > 0.0 {
                                let dep_value = (chain.cur.l / chain.cur.f)
                                    * dep_energy
                                    * (1.0 - accept_prob).clamp(0.0, 1.0);
                                indirect_film.add_sample(
                                    chain.cur.pixel,
                                    dep_value,
                                    chain.cur.lambda.clone(),
                                    1.0,
                                );
                            }
                            if accept_prob == 1.0 || rng.gen::<f32>() < accept_prob {
                                chain.cur = proposal;
                                chain.sampler.accept();
                            } else {
                                chain.sampler.reject();
                            }
                        }
                    }
                }
                acc_li += li;
                arena.reset();
            }
            acc_li = acc_li / (self.spp as f32);
            let _ = acc_li;
            // indirect_film.add_sample(uvec2(x, y), &acc_li, 1.0);
            if (id + 1) % 256 == 0 {
                progress.inc(1);
            }
        });
        progress.finish();
        let film = Film::new(&scene.camera.resolution());
        (0..npixels).into_par_iter().for_each(|i| {
            let mut px = film.pixels()[i].write();
            px.weight = RobustSum::new(1.0);
            {
                let px_i = indirect_film.pixels()[i].read();
                px.intensity.add(px_i.intensity.sum());
            }
            {
                let px_d = film_direct.pixels()[i].read();
                assert!(px_d.weight.sum() > 0.0);
                px.intensity.add(px_d.intensity.sum() / px_d.weight.sum());
            }
        });
        film
    }
}
