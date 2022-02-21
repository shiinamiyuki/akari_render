use std::sync::atomic::AtomicUsize;

use crate::distribution::Distribution1D;
use crate::sampler::{MltSampler, PCGSampler, Sampler};
use crate::scene::Scene;
use crate::util::PerThread;
use crate::*;
use bidir::*;
use bumpalo::Bump;
use film::Film;
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use UVec2;

use super::path::PathTracer;
use super::Integrator;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stream {
    Camera = 0,
    Light = 1,
    Connect = 2,
    TotalStreams = 3,
}

pub struct Mmlt {
    pub max_depth: u32,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    pub spp: u32,
    pub direct_spp: u32,
}
pub struct FRecord {
    pub pixel: UVec2,
    pub f: f32,
    pub l: SampledSpectrum,
    pub lambda: SampledWavelengths,
}

pub struct MmltSampler {
    streams: [MltSampler; 3],
    stream: Stream,
    large_step_prob: f32,
    large_step: bool,
    pub pcg: PCGSampler,
}
impl MmltSampler {
    fn new(seed: u64) -> Self {
        let mut pcg = PCGSampler::new(seed);
        Self {
            streams: [
                MltSampler::new(pcg.rng.pcg64()), // can the path be correlated?
                MltSampler::new(pcg.rng.pcg64()),
                MltSampler::new(pcg.rng.pcg64()),
            ],
            large_step: true,
            pcg,
            stream: Stream::Camera,
            large_step_prob: 0.3,
        }
    }
    fn use_stream(&mut self, stream: Stream) {
        self.stream = stream;
    }
    fn accept(&mut self) {
        for s in &mut self.streams {
            s.accept();
        }
    }
    fn reseed(&mut self, seed: u64) {
        self.pcg = PCGSampler::new(seed);
        for s in &mut self.streams {
            s.reseed(self.pcg.rng.pcg64())
        }
    }
    fn reject(&mut self) {
        for s in &mut self.streams {
            s.reject();
        }
    }
}
impl Sampler for MmltSampler {
    fn start_next_sample(&mut self) {
        self.large_step = self.pcg.next1d() < self.large_step_prob;
        for s in &mut self.streams {
            s.start_new_iteration(self.large_step);
        }
    }
    fn next1d(&mut self) -> f32 {
        self.streams[self.stream as usize].next1d()
    }
}

pub struct Chain {
    pub sampler: MmltSampler,
    pub cur: FRecord,
    pub depth: u32,
}
impl Chain {
    pub fn run_at_pixel(
        &mut self,
        pixel: UVec2,
        scene: &Scene,
        mut lambda: SampledWavelengths,
        arena: &Bump,
    ) -> FRecord {
        let (n_strategies, s, t) = if self.depth == 0 {
            (1, 0, 2)
        } else {
            let n_strategies = self.depth + 2;
            let s = ((self.sampler.next1d() * n_strategies as f32) as u32).min(n_strategies - 1)
                as usize;
            let t = self.depth as usize + 2 - s;
            (n_strategies, s, t)
        };
        let mut camera_path = Path::new(arena, self.depth as usize + 2);
        let mut light_path = Path::new(arena, self.depth as usize + 1);
        let mut new_camera_path = Path::new(arena, self.depth as usize + 2);
        let mut new_light_path = Path::new(arena, self.depth as usize + 1);
        bidir::generate_camera_path(
            scene,
            pixel,
            &mut self.sampler,
            &mut lambda,
            t,
            &mut camera_path,
            arena,
        );
        if camera_path.len() != t {
            return FRecord {
                pixel,
                f: 0.0,
                l: SampledSpectrum::zero(),
                lambda,
            };
        }
        self.sampler.use_stream(Stream::Light);
        bidir::generate_light_path(
            scene,
            &mut self.sampler,
            &mut lambda,
            s,
            &mut light_path,
            arena,
        );
        if light_path.len() != s {
            return FRecord {
                pixel,
                f: 0.0,
                l: SampledSpectrum::zero(),
                lambda,
            };
        }
        self.sampler.use_stream(Stream::Connect);
        let (l, w, raster) = bidir::connect_paths(
            scene,
            ConnectionStrategy { s, t },
            &light_path,
            &camera_path,
            &mut self.sampler,
            &mut lambda,
            &mut &mut new_light_path,
            &mut new_camera_path,
        );
        let l = l * w * n_strategies as f32;
        let l = if l.is_black() {
            SampledSpectrum::zero()
        } else {
            l
        };
        let pixel = if let Some(raster) = raster {
            raster
        } else {
            pixel
        };
        FRecord {
            pixel,
            f: lambda.clone().cie_xyz(l).values().y.clamp(0.0, 100.0),
            l,
            lambda,
        }
    }
    pub fn run(&mut self, scene: &Scene, arena: &Bump) -> FRecord {
        self.sampler.start_next_sample();
        self.sampler.use_stream(Stream::Camera);

        let pixel = self.sampler.next2d() * scene.camera.resolution().as_vec2();
        let pixel = uvec2(pixel.x as u32, pixel.y as u32);
        let pixel = uvec2(
            pixel.x.min(scene.camera.resolution().x - 1),
            pixel.y.min(scene.camera.resolution().y - 1),
        );
        let lambda = SampledWavelengths::sample_visible(self.sampler.pcg.next1d());
        self.run_at_pixel(pixel, scene, lambda, arena)
    }
}
impl Mmlt {
    pub fn init_chain(
        n_bootstrap: usize,
        n_chains: usize,
        depth: u32,
        scene: &Scene,
        arena: &Bump,
    ) -> (Vec<Chain>, f32) {
        let mut rng = thread_rng();
        let seeds: Vec<_> = { (0..n_bootstrap).map(|_| rng.gen::<u64>()).collect() };
        let fs: Vec<_> = (0..n_bootstrap)
            .map(|i| {
                Chain {
                    sampler: MmltSampler::new(seeds[i]),
                    cur: FRecord {
                        pixel: UVec2::ZERO,
                        f: 0.0,
                        l: SampledSpectrum::zero(),
                        lambda: SampledWavelengths::none(),
                    },
                    depth,
                }
                .run(scene, arena)
                .f
            })
            .collect();
        let dist = Distribution1D::new(&fs).unwrap_or_else(|| {
            panic!(
                "mmlt initialization failed for depth {}. increase n_bootstrap and try again",
                depth
            )
        });
        (
            (0..n_chains)
                .map(|_| {
                    let (i, _) = dist.sample_discrete(rng.gen::<f32>());
                    let mut chain = Chain {
                        sampler: MmltSampler::new(seeds[i]),
                        cur: FRecord {
                            pixel: UVec2::ZERO,
                            f: 0.0,
                            l: SampledSpectrum::zero(),
                            lambda: SampledWavelengths::none(),
                        },
                        depth,
                    };
                    chain.cur = chain.run(scene, arena);
                    chain.sampler.reseed(rng.gen());
                    chain
                })
                .collect(),
            fs.iter().sum::<f32>(),
        )
    }
}
impl Integrator for Mmlt {
    fn render(&self, scene: &scene::Scene) -> Film {
        log::info!("rendering direct lighting...");
        let mut depth0_pt = PathTracer {
            spp: self.direct_spp,
            max_depth: 1,
            single_wavelength:false,
        };
        let film_direct = depth0_pt.render(scene);
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        log::info!("bootstrapping...");
        let arenas = PerThread::new(|| Bump::new());

        let per_depth_chains: Vec<_> = (2..=self.max_depth)
            .into_par_iter()
            .map(|depth| {
                let arena = arenas.get_mut();
                let chain = Self::init_chain(self.n_bootstrap, self.n_chains, depth, scene, arena);
                arena.reset();
                chain
            })
            .collect();
        let per_depth_film: Vec<_> = per_depth_chains
            .iter()
            .map(|_| Film::new(&scene.camera.resolution()))
            .collect();
        let depth_fs: Vec<_> = per_depth_chains.iter().map(|x| x.1).collect();
        let depth_dist = Distribution1D::new(&depth_fs).unwrap();
        let depth_fs: Vec<_> = depth_fs.into_iter().map(|f| AtomicFloat::new(f)).collect();
        let per_depth_large: Vec<_> = depth_fs
            .iter()
            .map(|_| AtomicUsize::new(self.n_bootstrap))
            .collect();
        let per_depth_chains: Vec<_> = per_depth_chains
            .into_iter()
            .map(|x| {
                let chains: Vec<Mutex<Chain>> = x.0.into_iter().map(|x| Mutex::new(x)).collect();
                chains
            })
            .collect();

        {
            let bs: Vec<_> = depth_fs
                .iter()
                .zip(per_depth_large.iter())
                .map(|(f, n)| f.load(Ordering::SeqCst) as f64 / n.load(Ordering::SeqCst) as f64)
                .collect();
            log::info!("normalization factor inital estimate: {:?}", bs);
        }
        let progress = crate::util::create_progess_bar(self.spp as usize, "spp");
        log::info!("rendering {} spp", self.spp);
        for _ in 0..self.spp {
            (0..npixels).into_par_iter().for_each(|_| {
                let mut rng = thread_rng();
                let (depth, depth_pdf) = depth_dist.sample_discrete(rng.gen());
                let arena = arenas.get_mut();
                loop {
                    let chains = &per_depth_chains[depth];
                    let chain = &chains[rng.gen::<usize>() % chains.len()];
                    if let Some(mut chain) = chain.try_lock() {
                        let proposal = chain.run(scene, arena);
                        let accept_prob = match chain.cur.f {
                            x if x > 0.0 => (proposal.f / x).min(1.0),
                            _ => 1.0,
                        };
                        if chain.sampler.large_step {
                            depth_fs[depth].fetch_add(proposal.f, Ordering::SeqCst);
                            per_depth_large[depth].fetch_add(1, Ordering::Relaxed);
                        }
                        if proposal.f > 0.0 {
                            per_depth_film[depth].add_sample(
                                proposal.pixel,
                                proposal.l * accept_prob / (proposal.f * depth_pdf),
                                proposal.lambda.clone(),
                                1.0,
                            );
                        }
                        if chain.cur.f > 0.0 {
                            per_depth_film[depth].add_sample(
                                chain.cur.pixel,
                                chain.cur.l * (1.0 - accept_prob) / (chain.cur.f * depth_pdf),
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
                        break;
                    }
                }
                arena.reset();
            });
            progress.inc(1);
        }
        progress.finish();
        let bs: Vec<_> = depth_fs
            .iter()
            .zip(per_depth_large.iter())
            .map(|(f, n)| f.load(Ordering::SeqCst) as f64 / n.load(Ordering::SeqCst) as f64)
            .collect();
        log::info!("normalization factor: {:?}", bs);
        for (depth, film) in per_depth_film.iter().enumerate() {
            film.pixels().par_iter().enumerate().for_each(|(_, p)| {
                let mut px = p.write();
                px.intensity = RobustSum::new(px.intensity.sum() * bs[depth] as f32);
            });
        }
        let film = Film::new(&scene.camera.resolution());

        (0..npixels).into_par_iter().for_each(|i| {
            let mut px = film.pixels()[i].write();
            px.weight = RobustSum::new(1.0);
            for (_, film) in per_depth_film.iter().enumerate() {
                {
                    let px_d = film.pixels()[i].read();
                    px.intensity.add(px_d.intensity.sum() / self.spp as f32);
                }
            }
            {
                let px_d = film_direct.pixels()[i].read();
                assert!(px_d.weight.sum() > 0.0);
                px.intensity.add(px_d.intensity.sum() / px_d.weight.sum());
            }
        });
        // film.write_exr("mmlt.exr");
        film
    }
}
