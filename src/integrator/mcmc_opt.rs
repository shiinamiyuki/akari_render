use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;

use super::pt::{self, PathTracer};
use super::{Integrator, IntermediateStats, RenderOptions, RenderStats};
use crate::sampler::mcmc::{
    IsotropicExponentialMutation, KelemenMutationRecord, KelemenMutationRecordExpr, KELEMEN_MUTATE,
};
use crate::{
    color::*,
    film::*,
    sampler::{
        mcmc::{IsotropicGaussianMutation, LargeStepMutation, Mutation},
        *,
    },
    scene::*,
    util::alias_table::AliasTable,
    *,
};
use rand::Rng;
use serde::{Deserialize, Serialize};

use super::mcmc::{Config, Mcmc, Method};
#[derive(Clone, Copy, Value, Debug)]
#[repr(C)]
pub struct PssSample {
    pub cur: f32,
    pub backup: f32,
}
pub struct McmcOpt {
    pub device: Device,
    pub pt: PathTracer,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    pub mcmc_depth: u32,
    config: Config,
}

#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct MarkovState {
    chain_id: u32,
    cur_pixel: Uint2,
    cur_f: f32,
    b: f32,
    b_cnt: u64,
    n_accepted: u64,
    n_mutations: u64,
    sigma: f32,
}
struct RenderState {
    rng_states: Buffer<Pcg32>,
    samples: Buffer<PssSample>,
    states: Buffer<MarkovState>,
    cur_colors: ColorBuffer,
    b_init: f32,
    b_init_cnt: u32,
}

pub struct LazyMcmcSampler<'a> {
    pub base: &'a IndependentSampler,
    pub samples: BufferVar<PssSample>,
    pub offset: Expr<u32>,
    pub cur_dim: Var<u32>,
    pub mcmc_dim: Expr<u32>,
}
impl<'a> LazyMcmcSampler<'a> {
    pub fn new(
        base: &'a IndependentSampler,
        samples: BufferVar<PssSample>,
        offset: Expr<u32>,
        mcmc_dim: Expr<u32>,
    ) -> Self {
        Self {
            base,
            samples,
            offset,
            cur_dim: var!(u32, 0),
            mcmc_dim,
        }
    }
}

impl<'a> Sampler for LazyMcmcSampler<'a> {
    fn next_1d(&self) -> Float {
        if_!(self.cur_dim.load().cmplt(self.mcmc_dim), {
            let ret = self.samples.read(self.offset + self.cur_dim.load()).cur();
            self.cur_dim.store(self.cur_dim.load() + 1);
            ret
        }, else {
            self.cur_dim.store(self.cur_dim.load() + 1);
            self.base.next_1d()
        })
    }
    fn start(&self) {
        self.cur_dim.store(0);
    }
}
pub struct Mutator {
    pub method: Method,
    pub is_large_step: Expr<bool>,
}
impl Mutator {
    pub fn mutate(
        &self,
        samples: &Buffer<PssSample>,
        offset: Expr<u32>,
        dims: Expr<u32>,
        rng: &IndependentSampler,
    ) {
        match self.method {
            Method::Kelemen {
                exponential_mutation,
                small_sigma,
                large_step_prob: _,
                ..
            } => {
                let kelemen_mutate_size_low = const_(1.0 / 1024.0f32);
                let kelemen_mutate_size_high = const_(1.0 / 64.0f32);
                let kelemen_log_ratio = -(kelemen_mutate_size_high / kelemen_mutate_size_low).ln();
                for_range(const_(0i32)..dims.int(), |i| {
                    let i = i.uint();
                    let u = rng.next_1d();
                    let sample = var!(PssSample, samples.var().read(offset + i));
                    sample.set_backup(*sample.cur());
                    if_!(self.is_large_step, {
                        sample.set_cur(u);
                    }, else {
                        if exponential_mutation {
                            let record = var!(
                                KelemenMutationRecord,
                                KelemenMutationRecordExpr::new(
                                    *sample.cur(),
                                    u,
                                    kelemen_mutate_size_low,
                                    kelemen_mutate_size_high,
                                    kelemen_log_ratio,
                                    0.0
                                )
                            );
                            KELEMEN_MUTATE.call(record);
                            // cpu_dbg!(*record);
                            sample.set_cur(*record.mutated());
                        } else {
                            let tmp1 = (-2.0 * (1.0 - rng.next_1d()).ln()).sqrt();
                            let dv = tmp1 * (2.0 * PI * rng.next_1d()).cos();
                            let new = *sample.cur() + dv * small_sigma;
                            let new = new - new.floor();
                            let new = select(new.is_finite(), new, const_(0.0f32));
                            sample.set_cur(new);
                        }
                    });
                    samples.var().write(offset + i, *sample);
                });
            }
        }
    }
}
impl McmcOpt {
    fn sample_dimension(&self) -> usize {
        4 + self.mcmc_depth as usize * (3 + 3 + 1)
    }
    pub fn new(device: Device, config: Config) -> Self {
        let pt_config = pt::Config {
            spp: config.spp,
            max_depth: config.max_depth,
            spp_per_pass: config.spp_per_pass,
            use_nee: config.use_nee,
            rr_depth: config.rr_depth,
            indirect_only: config.direct_spp >= 0,
        };
        Self {
            device: device.clone(),
            pt: PathTracer::new(device.clone(), pt_config),
            method: config.method,
            n_chains: config.n_chains,
            n_bootstrap: config.n_bootstrap,
            mcmc_depth: config.mcmc_depth,
            config,
        }
    }
    fn evaluate(
        &self,
        scene: &Arc<Scene>,
        eval: &Evaluators,
        samples: &Buffer<PssSample>,
        independent: &IndependentSampler,
        chain_id: Expr<u32>,
    ) -> (Expr<Uint2>, Color, Expr<f32>) {
        let sampler = LazyMcmcSampler::new(
            independent,
            samples.var(),
            chain_id * const_(self.sample_dimension() as u32),
            const_(self.sample_dimension() as u32),
        );
        sampler.start();
        let res = const_(scene.camera.resolution());
        let p = sampler.next_2d() * res.float();
        let p = p.int().clamp(0, res.int() - 1);
        let (ray, ray_color, ray_w) =
            scene
                .camera
                .generate_ray(p.uint(), &sampler, eval.color_repr);
        let l = self.pt.radiance(scene, ray, &sampler, eval) * ray_color * ray_w;
        (p.uint(), l.clone(), Mcmc::scalar_contribution(&l))
    }
    fn bootstrap(&self, scene: &Arc<Scene>, eval: &Evaluators) -> RenderState {
        let seeds = init_pcg32_buffer(self.device.clone(), self.n_bootstrap + self.n_chains);
        let samples = self
            .device
            .create_buffer::<PssSample>(self.sample_dimension() * self.n_bootstrap);
        let fs = self
            .device
            .create_buffer_from_fn(self.n_bootstrap, |_| 0.0f32);
        self.device
            .create_kernel::<()>(&|| {
                let i = dispatch_id().x();
                let seed = seeds.var().read(i);
                let sampler = IndependentSampler {
                    state: var!(Pcg32, seed),
                };
                let dim = const_(self.sample_dimension() as u32);
                for_range(const_(0)..dim.int(), |j| {
                    let j = j.uint();
                    samples
                        .var()
                        .write(i * dim + j, PssSampleExpr::new(sampler.next_1d(), 0.0));
                });
                let (_p, _l, f) = self.evaluate(scene, eval, &samples, &sampler, i);
                fs.var().write(i, f);
            })
            .dispatch([self.n_bootstrap as u32, 1, 1]);

        let weights = fs.copy_to_vec();
        let b = weights.iter().sum::<f32>();
        assert!(b > 0.0, "Bootstrap failed, please retry with more samples");
        log::info!(
            "Normalization factor initial estimate: {}",
            b / self.n_bootstrap as f32
        );
        let at = AliasTable::new(self.device.clone(), &weights);
        let states = self.device.create_buffer(self.n_chains);
        let cur_colors = ColorBuffer::new(self.device.clone(), self.n_chains, eval.color_repr);
        let sample_buffer = self
            .device
            .create_buffer(self.sample_dimension() * self.n_chains);
        self.device
            .create_kernel::<()>(&|| {
                let i = dispatch_id().x();
                let seed = seeds.var().read(i + self.n_bootstrap as u32);
                let sampler = IndependentSampler {
                    state: var!(Pcg32, seed),
                };
                let (seed_idx, _, _) = at.sample_and_remap(sampler.next_1d());
                let seed = seeds.var().read(seed_idx);
                let sampler = IndependentSampler {
                    state: var!(Pcg32, seed),
                };
                let dim = const_(self.sample_dimension() as u32);
                for_range(const_(0)..dim.int(), |j| {
                    let j = j.uint();
                    sample_buffer
                        .var()
                        .write(i * dim + j, PssSampleExpr::new(sampler.next_1d(), 0.0));
                });

                let (p, l, f) = self.evaluate(scene, eval, &samples, &sampler, i);
                let sigma = match &self.method {
                    Method::Kelemen { small_sigma, .. } => *small_sigma,
                };
                cur_colors.write(i, l);
                let state = MarkovStateExpr::new(i, p, f, 0.0, 0, 0, 0, sigma);
                states.var().write(i, state);
            })
            .dispatch([self.n_chains as u32, 1, 1]);
        let rng_states = init_pcg32_buffer(self.device.clone(), self.n_chains);
        RenderState {
            rng_states,
            samples: sample_buffer,
            cur_colors,
            states,
            b_init: b,
            b_init_cnt: self.n_bootstrap as u32,
        }
    }
    fn mutate_chain(
        &self,
        scene: &Arc<Scene>,
        eval: &Evaluators,
        render_state: &RenderState,
        film: &Film,
        contribution: Expr<f32>,
        state: Var<MarkovState>,
        cur_color_v: ColorVar,
        rng: &IndependentSampler,
    ) {
        let offset = *state.chain_id() * const_(self.sample_dimension() as u32);
        // select a mutation strategy
        match self.method {
            Method::Kelemen {
                large_step_prob, ..
            } => {
                let is_large_step = rng.next_1d().cmplt(large_step_prob);
                let mutator = Mutator {
                    is_large_step,
                    method: self.method,
                };
                mutator.mutate(
                    &render_state.samples,
                    offset,
                    const_(self.sample_dimension() as u32),
                    rng,
                );
                let (proposal_p, proposal_color, f) =
                    self.evaluate(scene, eval, &render_state.samples, &rng, *state.chain_id());
                let proposal_f = f;
                if_!(is_large_step, {
                    state.set_b(state.b().load() + proposal_f);
                    state.set_b_cnt(state.b_cnt().load() + 1);
                });
                let cur_f = state.cur_f().load();
                let cur_p = state.cur_pixel().load();
                let cur_color = cur_color_v.load();
                let accept = select(
                    cur_f.cmpeq(0.0),
                    const_(1.0f32),
                    (proposal_f / cur_f).clamp(0.0, 1.0),
                );
                film.add_splat(
                    proposal_p.float(),
                    &(proposal_color.clone() / proposal_f),
                    accept * contribution,
                );
                film.add_splat(
                    cur_p.float(),
                    &(cur_color / cur_f),
                    (1.0 - accept) * contribution,
                );
                if_!(
                    rng.next_1d().cmplt(accept),
                    {
                        state.set_cur_f(proposal_f);
                        cur_color_v.store(proposal_color);
                        state.set_cur_pixel(proposal_p);
                        if_!(!is_large_step, {
                            state.set_n_accepted(state.n_accepted().load() + 1);
                        });
                    },
                    else, // reject
                    {
                        for_range(
                            const_(0)..const_(self.sample_dimension() as u32).int(),
                            |i| {
                                let i = i.uint();
                                let sample =
                                    var!(PssSample, render_state.samples.var().read(offset + i));
                                sample.set_cur(*sample.backup());
                                render_state.samples.var().write(offset + i, *sample);
                            },
                        );
                    }
                );
                if_!(!is_large_step, {
                    state.set_n_mutations(state.n_mutations().load() + 1);
                });
            }
            _ => todo!(),
        }
    }
    fn advance_chain(
        &self,
        scene: &Arc<Scene>,
        eval: &Evaluators,
        render_state: &RenderState,
        film: &Film,
        mutations_per_chain: Expr<u32>,
        contribution: Expr<f32>,
    ) {
        let i = dispatch_id().x();
        let markov_states = render_state.states.var();
        let sampler = IndependentSampler {
            state: var!(Pcg32, render_state.rng_states.var().read(i)),
        };
        let state = var!(MarkovState, markov_states.read(i));
        let cur_color_v = ColorVar::new(render_state.cur_colors.read(i));
        for_range(const_(0)..mutations_per_chain.int(), |_| {
            self.mutate_chain(
                scene,
                eval,
                render_state,
                film,
                contribution,
                state,
                cur_color_v,
                &sampler,
            );
        });
        render_state.cur_colors.write(i, cur_color_v.load());
        render_state.rng_states.var().write(i, sampler.state.load());
        markov_states.write(i, state.load());
    }

    fn render_loop(
        &self,
        scene: &Arc<Scene>,
        eval: &Evaluators,
        state: &RenderState,
        film: &mut Film,
        options: &RenderOptions,
    ) {
        let resolution = scene.camera.resolution();
        let npixels = resolution.x * resolution.y;

        let kernel = self.device.create_kernel::<(u32, f32)>(
            &|mutations_per_chain: Expr<u32>, contribution: Expr<f32>| {
                if is_cpu_backend() {
                    let num_threads = std::thread::available_parallelism().unwrap().get();
                    if self.n_chains <= num_threads * 20 {
                        set_block_size([1, 1, 1]);
                    } else {
                        set_block_size([256, 1, 1]);
                    }
                } else {
                    set_block_size([256, 1, 1]);
                }
                self.advance_chain(scene, eval, state, film, mutations_per_chain, contribution)
            },
        );
        let reconstruct = |film: &mut Film, spp: u32| {
            let states = state.states.copy_to_vec();
            let mut b = state.b_init as f64;
            let mut b_cnt = state.b_init_cnt as u64;
            let mut accepted = 0u64;
            let mut mutations = 0u64;
            for s in &states {
                b += s.b as f64;
                b_cnt += s.b_cnt;
                accepted += s.n_accepted;
                mutations += s.n_mutations;
            }
            let accept_rate = accepted as f64 / mutations as f64;
            let b = b / b_cnt as f64;
            log::info!("#indenpentent proposals: {}", b_cnt);
            log::info!("Normalization factor: {}", b);
            log::info!("Acceptance rate: {:.2}%", accept_rate * 100.0);
            film.set_splat_scale(b as f32 / spp as f32);
        };
        let mut acc_time = 0.0f64;
        let mut stats = RenderStats::default();
        {
            let mut cnt = 0;
            let spp_per_pass = self.pt.spp_per_pass;
            let progress = util::create_progess_bar(self.pt.spp as usize, "spp");
            while cnt < self.pt.spp {
                let tic = Instant::now();
                let cur_pass = (self.pt.spp - cnt).min(spp_per_pass);
                let n_mutations = npixels as u64 * cur_pass as u64;
                let mutations_per_chain = (n_mutations / self.n_chains as u64).max(1);
                if mutations_per_chain > u32::MAX as u64 {
                    panic!("Number of mutations per chain exceeds u32::MAX, please reduce spp per pass or increase number of chains");
                }
                let mutations_per_chain = mutations_per_chain as u32;
                // since mutations_per_chain is truncated, we need to compensate for the truncation error
                // mutations_per_chain * n_chains * contribution = n_mutations
                let contribution =
                    n_mutations as f32 / (mutations_per_chain as f32 * self.n_chains as f32);
                kernel.dispatch(
                    [self.n_chains as u32, 1, 1],
                    &mutations_per_chain,
                    &contribution,
                );
                progress.inc(cur_pass as u64);
                cnt += cur_pass;
                let toc = Instant::now();
                acc_time += toc.duration_since(tic).as_secs_f64();
                if options.save_intermediate {
                    let output_image: luisa::Tex2d<luisa::Float4> = self.device.create_tex2d(
                        luisa::PixelStorage::Float4,
                        scene.camera.resolution().x,
                        scene.camera.resolution().y,
                        1,
                    );
                    reconstruct(film, cnt);
                    film.copy_to_rgba_image(&output_image);
                    let path = format!("{}-{}.exr", options.session, cnt);
                    util::write_image(&output_image, &path);
                    stats.intermediate.push(IntermediateStats {
                        time: acc_time,
                        spp: cnt,
                        path,
                    });
                }
            }
            progress.finish();
            if options.save_stats {
                let file = File::create(format!("{}.json", options.session)).unwrap();
                let json = serde_json::to_value(&stats).unwrap();
                let writer = BufWriter::new(file);
                serde_json::to_writer(writer, &json).unwrap();
            }
        }

        log::info!("Rendering finished in {:.2}s", acc_time);
        reconstruct(film, self.pt.spp);
    }
}
impl Integrator for McmcOpt {
    fn render(&self, scene: Arc<Scene>, film: &mut Film, options: &RenderOptions) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig: {:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        let color_repr = ColorRepr::Rgb;
        let evaluators = scene.evaluators(color_repr);
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        if self.config.direct_spp > 0 {
            log::info!(
                "Rendering direct illumination: {} spp",
                self.config.direct_spp
            );
            let direct = PathTracer::new(
                self.device.clone(),
                pt::Config {
                    max_depth: 2,
                    rr_depth: 2,
                    spp: self.config.direct_spp as u32,
                    indirect_only: false,
                    spp_per_pass: self.pt.spp_per_pass,
                    use_nee: self.pt.use_nee,
                },
            );
            direct.render(scene.clone(), film, &Default::default());
        }
        let render_state = self.bootstrap(&scene, &evaluators);
        self.render_loop(&scene, &evaluators, &render_state, film, options);
    }
}

pub fn render(
    device: Device,
    scene: Arc<Scene>,
    film: &mut Film,
    config: &Config,
    options: &RenderOptions,
) {
    let mcmc = McmcOpt::new(device.clone(), config.clone());
    mcmc.render(scene, film, options);
}
