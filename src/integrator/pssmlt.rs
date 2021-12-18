use glm::UVec2;
use parking_lot::Mutex;
use rand::{thread_rng, Rng};

use super::path::PathTracer;
use super::{mmlt::FRecord, Integrator};
use crate::distribution::Distribution1D;
use crate::film::Film;
use crate::sampler::Sampler;
use crate::*;
use crate::{sampler::MltSampler, scene::Scene};
pub struct Chain {
    pub sampler: MltSampler,
    pub cur: FRecord,
    pub is_large_step: bool,
    pub large_step_prob: f32,
    pub max_depth: usize,
}

pub struct Pssmlt {
    pub max_depth: u32,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    pub spp: u32,
    pub direct_spp: u32,
}
impl Chain {
    pub fn run_at_pixel(&mut self, pixel: &UVec2, scene: &Scene) -> FRecord {
        let (ray, _) = scene.camera.generate_ray(pixel, &mut self.sampler);
        let l = PathTracer::li(ray, &mut self.sampler, scene, self.max_depth, true);
        FRecord {
            pixel: *pixel,
            f: glm::comp_max(&l.samples).clamp(0.0, 100.0),
            l,
        }
    }
    pub fn run(&mut self, scene: &Scene) -> FRecord {
        self.is_large_step = self.sampler.rng.next1d() < self.large_step_prob;
        self.sampler.start_new_iteration(self.is_large_step);

        let pixel = self
            .sampler
            .next2d()
            .component_mul(&scene.camera.resolution().cast::<Float>());
        let pixel = uvec2(pixel.x as u32, pixel.y as u32);
        let pixel = uvec2(
            pixel.x.min(scene.camera.resolution().x - 1),
            pixel.y.min(scene.camera.resolution().y - 1),
        );
        self.run_at_pixel(&pixel, scene)
    }
}
impl Pssmlt {
    pub fn init_chain(
        max_depth: usize,
        n_bootstrap: usize,
        n_chains: usize,
        scene: &Scene,
    ) -> (Vec<Chain>, Float) {
        let mut rng = thread_rng();
        let seeds: Vec<_> = { (0..n_bootstrap).map(|_| rng.gen::<u64>()).collect() };
        let fs: Vec<_> = (0..n_bootstrap)
            .map(|i| {
                Chain {
                    sampler: MltSampler::new(seeds[i]),
                    cur: FRecord {
                        pixel: glm::zero(),
                        f: 0.0,
                        l: Spectrum::zero(),
                    },
                    is_large_step: true,
                    large_step_prob: 0.3,
                    max_depth,
                }
                .run(scene)
                .f
            })
            .collect();
        let dist = Distribution1D::new(&fs).unwrap_or_else(|| {
            panic!("pssmlt initialization failed. increase n_bootstrap and try again",)
        });
        (
            (0..n_chains)
                .map(|_| {
                    let (i, _) = dist.sample_discrete(rng.gen::<Float>());
                    let mut chain = Chain {
                        sampler: MltSampler::new(seeds[i]),
                        cur: FRecord {
                            pixel: glm::zero(),
                            f: 0.0,
                            l: Spectrum::zero(),
                        },
                        is_large_step: true,
                        large_step_prob: 0.3,
                        max_depth,
                    };
                    chain.cur = chain.run(scene);
                    chain.sampler.reseed(rng.gen());
                    chain
                })
                .collect(),
            fs.iter().sum::<Float>(),
        )
    }
}
impl Integrator for Pssmlt {
    fn render(&mut self, scene: &crate::scene::Scene) -> Film {
        log::info!("rendering direct lighting...");
        let mut depth0_pt = PathTracer {
            spp: self.direct_spp,
            max_depth: 1,
        };
        let film_direct = depth0_pt.render(scene);
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        log::info!("bootstrapping...");
        let (chains, b) = Self::init_chain(
            self.max_depth as usize,
            self.n_bootstrap,
            self.n_chains,
            scene,
        );
        let chains: Vec<Mutex<Chain>> = chains.into_iter().map(|x| Mutex::new(x)).collect();
        log::info!("normalization factor inital estimate: {}", b /  self.n_bootstrap as f32);
        let b = AtomicFloat::new(b);
        let large_count = AtomicUsize::new(self.n_bootstrap);
        let progress = crate::util::create_progess_bar(self.spp as usize, "spp");
        log::info!("rendering {} spp", self.spp);
        let indirect_film = Film::new(&scene.camera.resolution());
        for _ in 0..self.spp {
            (0..npixels).into_par_iter().for_each(|_| {
                let mut rng = thread_rng();
                loop {
                    let chain = &chains[rng.gen::<usize>() % chains.len()];
                    if let Some(mut chain) = chain.try_lock() {
                        let proposal = chain.run(scene);
                        let accept_prob = match chain.cur.f {
                            x if x > 0.0 => (proposal.f / x).min(1.0),
                            _ => 1.0,
                        };
                        if chain.sampler.large_step {
                            b.fetch_add(proposal.f, Ordering::SeqCst);
                            large_count.fetch_add(1, Ordering::Relaxed);
                        }
                        if proposal.f > 0.0 {
                            indirect_film.add_sample(
                                &proposal.pixel,
                                &(proposal.l * accept_prob / proposal.f),
                                1.0,
                            );
                        }
                        if chain.cur.f > 0.0 {
                            indirect_film.add_sample(
                                &chain.cur.pixel,
                                &(chain.cur.l * (1.0 - accept_prob) / chain.cur.f),
                                1.0,
                            );
                        }
                        if accept_prob == 1.0 || rng.gen::<Float>() < accept_prob {
                            chain.cur = proposal;
                            chain.sampler.accept();
                        } else {
                            chain.sampler.reject();
                        }
                        break;
                    }
                }
            });
            progress.inc(1);
        }
        progress.finish();
        let b = b.load(Ordering::Relaxed) as f64 / large_count.load(Ordering::Relaxed) as f64;
        log::info!("normalization factor: {:?}", b);
        indirect_film
            .pixels
            .par_iter()
            .enumerate()
            .for_each(|(_, p)| {
                let mut px = p.write();
                px.intensity = px.intensity * b as Float;
            });
        let film = Film::new(&scene.camera.resolution());
        (0..npixels).into_par_iter().for_each(|i| {
            let mut px = film.pixels[i].write();
            px.weight = 1.0;
            {
                let px_i = indirect_film.pixels[i].read();
                px.intensity += px_i.intensity / self.spp as Float;
            }
            {
                let px_d = film_direct.pixels[i].read();
                assert!(px_d.weight > 0.0);
                px.intensity += px_d.intensity / px_d.weight;
            }
        });
        film
    }
}
