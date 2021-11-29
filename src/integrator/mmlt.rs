use std::convert::TryInto;
use std::sync::atomic::AtomicUsize;

use crate::distribution::Distribution1D;
use crate::sampler::{PCGSampler, Sampler, PCG};
use crate::scene::Scene;
use crate::util::erf_inv;
use crate::*;
use bidir::*;
use film::Film;
use glm::UVec2;
use parking_lot::Mutex;
use rand::{thread_rng, Rng};

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

pub struct PrimarySample {
    value: Float,
    backup: Float,
    last_modified: usize,
    modified_backup: usize,
}
impl PrimarySample {
    pub fn new(x: Float) -> Self {
        Self {
            value: x,
            backup: 0.0,
            last_modified: 0,
            modified_backup: 0,
        }
    }
    pub fn backup(&mut self) {
        self.backup = self.value;
        self.modified_backup = self.last_modified;
    }
    pub fn restore(&mut self) {
        self.value = self.backup;
        self.last_modified = self.modified_backup;
    }
}
pub struct MLTSampler {
    stream: Stream,
    samples: Vec<PrimarySample>,
    rng: PCGSampler,
    large_step: bool,
    cur_iteration: isize,
    large_step_prob: Float,
    dimension: usize,
    last_large_iteration: usize,
}
const SIGMA: Float = 0.01;
impl MLTSampler {
    pub fn new(seed: usize) -> Self {
        Self {
            stream: Stream::Camera,
            samples: vec![],
            rng: PCGSampler {
                rng: PCG::new(seed),
            },
            large_step: false,
            large_step_prob: 0.3,
            last_large_iteration: 0,
            cur_iteration: 0,
            dimension: 0,
        }
    }
    fn use_stream(&mut self, stream: Stream) {
        self.stream = stream;
        self.dimension = 0;
    }
    fn update(&mut self, i: usize) {
        let x = &mut self.samples[i];
        if x.last_modified < self.last_large_iteration {
            x.value = self.rng.next1d();
            x.last_modified = self.last_large_iteration;
        }
        x.backup();
        if self.large_step {
            x.value = self.rng.next1d();
        } else {
            let n_small: usize = self.cur_iteration as usize - x.last_modified;
            let normal_sample = (2.0 as Float).sqrt() * erf_inv(2.0 * self.rng.next1d() - 1.0);
            let err_sigma = SIGMA * (n_small as Float).sqrt();
            x.value += normal_sample * err_sigma;
            x.value -= x.value.floor();
        }
        x.last_modified = self.cur_iteration.try_into().unwrap();
    }
    fn accept(&mut self) {
        if self.large_step {
            self.last_large_iteration = self.cur_iteration as usize;
        }
    }
    fn reject(&mut self) {
        for x in &mut self.samples {
            if x.last_modified == self.cur_iteration as usize {
                x.restore();
            }
        }
        self.cur_iteration -= 1;
    }
}
impl Sampler for MLTSampler {
    fn start_next_sample(&mut self) {
        self.cur_iteration += 1;
        self.large_step = self.rng.next1d() < self.large_step_prob;
        self.dimension = 0;
    }

    fn next1d(&mut self) -> Float {
        let idx = self.dimension * Stream::TotalStreams as usize + self.stream as usize;
        self.dimension += 1;
        while idx >= self.samples.len() {
            self.samples.push(PrimarySample::new(self.rng.next1d()));
        }
        self.update(idx);
        self.samples[idx].value
    }
}

pub struct MMLT {
    pub(crate) max_depth: u32,
    pub(crate) n_chains: usize,
    pub(crate) n_bootstrap: usize,
    pub(crate) spp: u32,
    pub(crate) direct_spp: u32,
}
pub struct FRecord {
    pixel: UVec2,
    f: Float,
    l: Spectrum,
}

struct Chain<'a> {
    sampler: MLTSampler,
    cur: FRecord,
    depth: u32,
    light_path: Vec<Vertex<'a>>,
    camera_path: Vec<Vertex<'a>>,
    scratch: Scratch<'a>,
}
impl<'a> Chain<'a> {
    fn run(&mut self, scene: &'a Scene) -> FRecord {
        self.sampler.start_next_sample();
        self.sampler.use_stream(Stream::Camera);
        self.light_path.clear();
        self.camera_path.clear();
        self.scratch.new_light_path.clear();
        self.scratch.new_eye_path.clear();

        let (n_strategies, s, t) = if self.depth == 0 {
            (1, 0, 2)
        } else {
            let n_strategies = self.depth + 1;
            let s = ((self.sampler.next1d() * n_strategies as Float) as u32).min(n_strategies - 1)
                as usize;
            let t = self.depth as usize + 2 - s;
            (n_strategies, s, t)
        };
        let pixel = self
            .sampler
            .next2d()
            .component_mul(&scene.camera.resolution().cast::<Float>());
        let pixel = uvec2(pixel.x as u32, pixel.y as u32);
        let pixel = uvec2(
            pixel.x.min(scene.camera.resolution().x - 1),
            pixel.y.min(scene.camera.resolution().y - 1),
        );
        bidir::generate_camera_path(scene, &pixel, &mut self.sampler, t, &mut self.camera_path);
        if self.camera_path.len() != t {
            return FRecord {
                pixel,
                f: 0.0,
                l: Spectrum::zero(),
            };
        }
        self.sampler.use_stream(Stream::Light);
        bidir::generate_light_path(scene, &mut self.sampler, s, &mut self.light_path);
        if self.light_path.len() != s {
            return FRecord {
                pixel,
                f: 0.0,
                l: Spectrum::zero(),
            };
        }
        self.sampler.use_stream(Stream::Connect);
        let l = bidir::connect_paths(
            scene,
            ConnectionStrategy { s, t },
            &self.light_path,
            &self.camera_path,
            &mut self.sampler,
            &mut self.scratch,
        ) * n_strategies as Float;
        let l = if l.is_black() { Spectrum::zero() } else { l };
        FRecord {
            pixel,
            f: glm::comp_max(&l.samples).clamp(0.0, 100.0),
            l,
        }
    }
}
impl MMLT {
    fn init_chain<'a>(
        n_bootstrap: usize,
        n_chains: usize,
        depth: u32,
        scene: &'a Scene,
    ) -> (Vec<Chain<'a>>, Float) {
        let mut rng = thread_rng();
        let seeds: Vec<_> = { (0..n_bootstrap).map(|_| rng.gen::<usize>()).collect() };
        let fs: Vec<_> = (0..n_bootstrap)
            .map(|i| {
                Chain {
                    sampler: MLTSampler::new(seeds[i]),
                    cur: FRecord {
                        pixel: glm::zero(),
                        f: 0.0,
                        l: Spectrum::zero(),
                    },
                    light_path: vec![],
                    camera_path: vec![],
                    depth,
                    scratch: Scratch::new(),
                }
                .run(scene)
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
                    let (i, _) = dist.sample_discrete(rng.gen::<Float>());
                    let mut chain = Chain {
                        sampler: MLTSampler::new(seeds[i]),
                        cur: FRecord {
                            pixel: glm::zero(),
                            f: 0.0,
                            l: Spectrum::zero(),
                        },
                        light_path: vec![],
                        camera_path: vec![],
                        depth,
                        scratch: Scratch::new(),
                    };
                    chain.cur = chain.run(scene);
                    chain.sampler.rng = PCGSampler {
                        rng: PCG::new(rng.gen()),
                    };
                    chain
                })
                .collect(),
            fs.iter().sum::<Float>(),
        )
    }
}
impl Integrator for MMLT {
    fn render(&mut self, scene: &scene::Scene) -> Film {
        log::info!("rendering direct lighting...");
        let mut depth0_pt = PathTracer {
            spp: self.direct_spp,
            max_depth: 1,
        };
        let film_direct = depth0_pt.render(scene);
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        log::info!("bootstrapping...");
        let per_depth_chains: Vec<_> = (2..=self.max_depth)
            .into_par_iter()
            .map(|depth| Self::init_chain(self.n_bootstrap, self.n_chains, depth, scene))
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
                loop {
                    let chains = &per_depth_chains[depth];
                    let chain = &chains[rng.gen::<usize>() % chains.len()];
                    if let Some(mut chain) = chain.try_lock() {
                        let proposal = chain.run(scene);
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
                                &proposal.pixel,
                                &(proposal.l * accept_prob / (proposal.f * depth_pdf)),
                                1.0,
                            );
                        }
                        if chain.cur.f > 0.0 {
                            per_depth_film[depth].add_sample(
                                &chain.cur.pixel,
                                &(chain.cur.l * (1.0 - accept_prob) / (chain.cur.f * depth_pdf)),
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
        let bs: Vec<_> = depth_fs
            .iter()
            .zip(per_depth_large.iter())
            .map(|(f, n)| f.load(Ordering::SeqCst) as f64 / n.load(Ordering::SeqCst) as f64)
            .collect();
        log::info!("normalization factor: {:?}", bs);
        for (depth, film) in per_depth_film.iter().enumerate() {
            film.pixels.par_iter().enumerate().for_each(|(_, p)| {
                let mut px = p.write();
                px.intensity = px.intensity * bs[depth] as Float;
            });
        }
        let film = Film::new(&scene.camera.resolution());

        (0..npixels).into_par_iter().for_each(|i| {
            let mut px = film.pixels[i].write();
            px.weight = 1.0;
            for (_, film) in per_depth_film.iter().enumerate() {
                {
                    let px_d = film.pixels[i].read();
                    px.intensity += px_d.intensity / self.spp as Float;
                }
            }
            {
                let px_d = film_direct.pixels[i].read();
                assert!(px_d.weight > 0.0);
                px.intensity += px_d.intensity / px_d.weight;
            }
        });
        // film.write_exr("mmlt.exr");
        film
    }
}
