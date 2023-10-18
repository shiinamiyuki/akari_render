use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;

use luisa::lang::debug::{comment, is_cpu_backend};

use super::pt::{self, PathTracer};
use super::{Integrator, IntermediateStats, RenderSession, RenderStats};
use crate::sampler::mcmc::{mutate_image_space_single, KelemenMutationRecord, KELEMEN_MUTATE};
use crate::sampling::sample_gaussian;
use crate::util::distribution::resample_with_f64;
use crate::{color::*, film::*, sampler::*, scene::*, *};

use super::mcmc::{Config, Method};
pub type PssBuffer = Buffer<PssSample>;
#[derive(Clone, Copy, Value, Debug, Soa)]
#[repr(C, align(16))]
#[value_new(pub)]
pub struct PssSample {
    pub cur: f32,
    pub backup: f32,
    pub last_modified: u32,
    pub modified_backup: u32,
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
#[value_new(pub)]
pub struct MarkovState {
    cur_pixel: Uint2,
    chain_id: u32,
    cur_f: f32,
    b: f32,
    b_cnt: u32,
    n_accepted: u32,
    n_mutations: u32,
    cur_iter: u32,
    last_large_iter: u32,
}
struct RenderState {
    rng_states: Buffer<Pcg32>,
    samples: PssBuffer,
    states: Buffer<MarkovState>,
    cur_colors: ColorBuffer,
    b_init: f64,
    b_init_cnt: usize,
}

pub struct LazyMcmcSampler<'a> {
    pub base: &'a IndependentSampler,
    pub samples: &'a PssBuffer,
    pub offset: Expr<u32>,
    pub cur_dim: Var<u32>,
    pub mcmc_dim: Expr<u32>,
    pub mutator: Option<Mutator>,
}
impl<'a> LazyMcmcSampler<'a> {
    pub fn new(
        base: &'a IndependentSampler,
        samples: &'a PssBuffer,
        offset: Expr<u32>,
        mcmc_dim: Expr<u32>,
        mutator: Option<Mutator>,
    ) -> Self {
        Self {
            base,
            samples,
            offset,
            cur_dim: 0u32.var(),
            mcmc_dim,
            mutator,
        }
    }
}

impl<'a> Sampler for LazyMcmcSampler<'a> {
    #[tracked]
    fn next_1d(&self) -> Expr<f32> {
        if self.cur_dim.load().lt(self.mcmc_dim) {
            let ret = if let Some(m) = &self.mutator {
                m.mutate_one(self.samples, self.offset, **self.cur_dim, self.base)
                    .cur
            } else {
                self.samples.var().read(self.offset + **self.cur_dim).cur
            };
            *self.cur_dim += 1;
            ret
        } else {
            *self.cur_dim += 1;
            self.base.next_1d()
        }
    }
    fn is_metropolis(&self) -> bool {
        false
    }
    fn uniform(&self) -> Expr<f32> {
        self.base.next_1d()
    }
    fn start(&self) {
        self.cur_dim.store(0);
    }
    fn clone_box(&self) -> Box<dyn Sampler> {
        todo!()
    }
    fn forget(&self) {
        todo!()
    }
}
pub struct Mutator {
    pub method: Method,
    pub is_large_step: Expr<bool>,
    pub is_image_mutation: Expr<bool>,
    pub last_large_iter: Expr<u32>,
    pub cur_iter: Expr<u32>,
    pub res: Expr<Float2>,
}
impl Mutator {
    #[tracked]
    pub fn mutate_one(
        &self,
        samples: &PssBuffer,
        offset: Expr<u32>,
        i: Expr<u32>, // dim
        rng: &IndependentSampler,
    ) -> Expr<PssSample> {
        match self.method {
            Method::Kelemen {
                exponential_mutation,
                small_sigma,
                image_mutation_size,
                image_mutation_prob,
                ..
            } => {
                let kelemen_mutate_size_low = 1.0 / 1024.0f32;
                let kelemen_mutate_size_high = 1.0 / 64.0f32;
                let kelemen_log_ratio = -(kelemen_mutate_size_high / kelemen_mutate_size_low).ln();
                let ret = Var::<PssSample>::zeroed();
                maybe_outline(|| {
                    let sample = samples.var().read(offset + i).var();
                    comment("mcmc mutate_one");
                    let u = rng.next_1d();
                    if sample.last_modified.lt(self.last_large_iter) {
                        *sample.cur = rng.next_1d();
                        *sample.last_modified = self.last_large_iter;
                    };

                    *sample.backup = sample.cur;
                    *sample.modified_backup = sample.last_modified;
                    if self.is_large_step {
                        *sample.cur = u;
                    } else {
                        let is_cur_dim_under_image_mutation =
                            image_mutation_size.is_some() & self.is_image_mutation;
                        let should_cur_dim_be_mutated = !is_cur_dim_under_image_mutation | i.lt(2);
                        let target_iter = if should_cur_dim_be_mutated {
                            self.cur_iter
                        } else {
                            self.cur_iter - 1
                        };
                        lc_assert!(target_iter.ge(sample.last_modified));
                        let n_small = target_iter - sample.last_modified;
                        if exponential_mutation {
                            let x = sample.cur.var();
                            for_range(0u32.expr()..n_small, |_| {
                                let u = rng.next_1d();
                                if u.lt(1.0 - image_mutation_prob) {
                                    let u = u / (1.0 - image_mutation_prob);
                                    let record = KelemenMutationRecord::new_expr(
                                        **x,
                                        u,
                                        kelemen_mutate_size_low,
                                        kelemen_mutate_size_high,
                                        kelemen_log_ratio,
                                        0.0,
                                    )
                                    .var();
                                    KELEMEN_MUTATE.call(record);
                                    x.store(**record.mutated);
                                };
                            });
                            *sample.cur = x;
                        } else {
                            if n_small.gt(0) {
                                // let tmp1 = (-2.0 * (1.0 - rng.next_1d()).ln()).sqrt();
                                // let dv = tmp1 * (2.0 * PI * rng.next_1d()).cos();
                                let dv = sample_gaussian(u);
                                let new = sample.cur
                                    + dv * small_sigma
                                        * ((1.0 - image_mutation_prob) * n_small.cast_f32()).sqrt();
                                let new = new - new.floor();
                                let new = select(new.is_finite(), new, 0.0f32.expr());
                                *sample.cur = new;
                            };
                        };
                        if image_mutation_size.is_some() {
                            if self.is_image_mutation & i.lt(2) {
                                let new = mutate_image_space_single(
                                    **sample.cur,
                                    rng,
                                    image_mutation_size.unwrap().expr(),
                                    self.res,
                                    i,
                                );
                                *sample.cur = new;
                            };
                        }
                    };
                    *sample.last_modified = self.cur_iter;
                    samples.var().write(offset + i, **sample);
                    *ret = sample;
                });
                **ret
            }
        }
    }
}
impl McmcOpt {
    fn sample_dimension(&self) -> usize {
        4 + 1 + (1 + self.mcmc_depth as usize) * (3 + 3 + 1)
    }
    pub fn new(device: Device, config: Config) -> Self {
        let pt_config = pt::Config {
            spp: config.spp,
            max_depth: config.max_depth,
            spp_per_pass: config.spp_per_pass,
            use_nee: config.use_nee,
            rr_depth: config.rr_depth,
            indirect_only: config.direct_spp >= 0,
            ..Default::default()
        };
        Self {
            device: device.clone(),
            pt: PathTracer::new(device.clone(), pt_config),
            method: config.method,
            n_chains: config.n_chains,
            n_bootstrap: config.n_bootstrap,
            mcmc_depth: config.mcmc_depth.unwrap_or(pt_config.max_depth),
            config,
        }
    }
    #[tracked]
    fn evaluate(
        &self,
        scene: &Arc<Scene>,
        filter: PixelFilter,
        color_pipeline: ColorPipeline,
        samples: &PssBuffer,
        independent: &IndependentSampler,
        chain_id: Expr<u32>,
        mutator: Option<Mutator>,
        is_bootstrap: bool,
    ) -> (
        Expr<Uint2>,
        Color,
        Expr<SampledWavelengths>,
        Expr<f32>,
        Expr<u32>,
    ) {
        let sampler = LazyMcmcSampler::new(
            independent,
            samples,
            chain_id * (self.sample_dimension() as u32).expr(),
            if is_bootstrap {
                0u32
            } else {
                self.sample_dimension() as u32
            }
            .expr(),
            mutator,
        );
        sampler.start();
        let res = scene.camera.resolution().expr();
        let p = sampler.next_2d() * res.cast_f32();
        let p = p.cast_i32().clamp(0, res.cast_i32() - 1);
        let swl = sample_wavelengths(color_pipeline.color_repr, &sampler).var();
        let (ray, ray_w) = scene.camera.generate_ray(
            &scene,
            filter,
            p.cast_u32(),
            &sampler,
            color_pipeline.color_repr,
            **swl,
        );
        let l = self.pt.radiance(scene, color_pipeline, ray, swl, &sampler) * ray_w;
        (
            p.cast_u32(),
            l,
            **swl,
            Self::scalar_contribution(&l),
            **sampler.cur_dim,
        )
    }
    pub fn scalar_contribution(color: &Color) -> Expr<f32> {
        color.max().clamp(0.0f32.expr(), 1e5f32.expr())
        // 1.0f32.expr()
    }
    fn bootstrap(
        &self,
        scene: &Arc<Scene>,
        filter: PixelFilter,
        color_pipeline: ColorPipeline,
    ) -> RenderState {
        let seeds = init_pcg32_buffer_with_seed(self.device.clone(), self.n_bootstrap, 0);
        let fs = self
            .device
            .create_buffer_from_fn(self.n_bootstrap, |_| 0.0f32);
        let sample_buffer = self
            .device
            .create_buffer(self.sample_dimension() * self.n_chains);
        {
            let pss_samples =
                self.sample_dimension() * self.n_chains * std::mem::size_of::<PssSample>();
            let states = self.n_chains * std::mem::size_of::<MarkovState>();
            log::info!(
                "Mcmc memory consumption {:.2}MiB: PSS samples: {:.2}MiB, Markov states: {:.2}MiB",
                (pss_samples + states) as f64 / 1024.0 / 1024.0,
                pss_samples as f64 / 1024.0 / 1024.0,
                states as f64 / 1024.0 / 1024.0
            );
        }
        self.device
            .create_kernel::<fn()>(&|| {
                let i = dispatch_id().x;
                let seed = seeds.var().read(i);
                let sampler = IndependentSampler::from_pcg32(seed.var());
                // DON'T WRITE INTO sample_buffer
                let (_p, _l, _swl, f, _) = self.evaluate(
                    scene,
                    filter,
                    color_pipeline,
                    &sample_buffer,
                    &sampler,
                    i,
                    None,
                    true,
                );
                fs.var().write(i, f);
            })
            .dispatch([self.n_bootstrap as u32, 1, 1]);

        let weights = fs.copy_to_vec();
        let (b, resampled) = resample_with_f64(&weights, self.n_chains);
        assert!(b > 0.0, "Bootstrap failed, please retry with more samples");
        log::info!(
            "Normalization factor initial estimate: {}",
            b / self.n_bootstrap as f64
        );
        let resampled = self.device.create_buffer_from_slice(&resampled);
        let states = self.device.create_buffer(self.n_chains);
        let cur_colors = ColorBuffer::new(
            self.device.clone(),
            self.n_chains,
            color_pipeline.color_repr,
        );

        self.device
            .create_kernel::<fn()>(&track!(|| {
                let i = dispatch_id().x;
                let seed_idx = resampled.var().read(i);
                let seed = seeds.var().read(seed_idx);
                let sampler = IndependentSampler::from_pcg32(seed.var());
                let dim = (self.sample_dimension() as u32).expr();
                for_range(0u32.expr()..dim, |j| {
                    sample_buffer.var().write(
                        i * dim + j,
                        PssSample::new_expr(sampler.next_1d(), 0.0, 0, 0),
                    );
                });

                let (p, l, swl, f, _) = self.evaluate(
                    scene,
                    filter,
                    color_pipeline,
                    &sample_buffer,
                    &sampler,
                    i,
                    None,
                    false,
                );
                cur_colors.write(i, l, swl);
                let state = MarkovState::new_expr(p, i, f, 0.0, 0, 0, 0, 0, 0);
                states.var().write(i, state);
            }))
            .dispatch([self.n_chains as u32, 1, 1]);
        let rng_states = init_pcg32_buffer_with_seed(self.device.clone(), self.n_chains, 1);
        RenderState {
            rng_states,
            samples: sample_buffer,
            cur_colors,
            states,
            b_init: b,
            b_init_cnt: self.n_bootstrap,
        }
    }
    #[tracked]
    fn mutate_chain(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        render_state: &RenderState,
        film: &Film,
        contribution: Expr<f32>,
        state: Var<MarkovState>,
        cur_color_v: ColorVar,
        cur_swl: Var<SampledWavelengths>,
        rng: &IndependentSampler,
    ) {
        let offset = state.chain_id * self.sample_dimension() as u32;
        *state.cur_iter += 1;
        // select a mutation strategy
        match self.method {
            Method::Kelemen {
                large_step_prob,
                image_mutation_prob,
                ..
            } => {
                let u = rng.next_1d();
                let is_large_step = u.lt(large_step_prob);
                let is_image_mutation = rng.next_1d().lt(image_mutation_prob);
                let mutator = Mutator {
                    is_large_step,
                    is_image_mutation,
                    method: self.method,
                    last_large_iter: **state.last_large_iter,
                    cur_iter: **state.cur_iter,
                    res: scene.camera.resolution().expr().cast_f32(),
                };
                let (proposal_p, proposal_color, proposal_swl, f, proposal_dim) = self.evaluate(
                    scene,
                    film.filter(),
                    color_pipeline,
                    &render_state.samples,
                    &rng,
                    **state.chain_id,
                    Some(mutator),
                    false,
                );
                let proposal_f = f;
                if is_large_step & state.b_cnt.lt(1024u32 * 1024) {
                    *state.b += proposal_f;
                    *state.b_cnt += 1;
                };
                let cur_f = **state.cur_f;
                let cur_p = **state.cur_pixel;
                let cur_color = cur_color_v.load();
                let accept = select(
                    proposal_f.is_finite(),
                    select(
                        cur_f.eq(0.0) | !cur_f.is_finite(),
                        1.0f32.expr(),
                        (proposal_f / cur_f).clamp(0.0f32.expr(), 1.0f32.expr()),
                    ),
                    0.0f32.expr(),
                );
                film.add_splat(
                    proposal_p.cast_f32(),
                    &(proposal_color.clone() / proposal_f),
                    proposal_swl,
                    accept * contribution,
                );
                film.add_splat(
                    cur_p.cast_f32(),
                    &(cur_color / cur_f),
                    **cur_swl,
                    (1.0 - accept) * contribution,
                );
                if rng.next_1d().lt(accept) {
                    *state.cur_f = proposal_f;
                    cur_color_v.store(proposal_color);
                    cur_swl.store(proposal_swl);
                    *state.cur_pixel = proposal_p;
                    if !is_large_step {
                        *state.n_accepted += 1;
                    } else {
                        *state.last_large_iter = state.cur_iter;
                    };
                } else
                // reject
                {
                    *state.cur_iter -= 1;
                    let dim = proposal_dim.min_(self.sample_dimension() as u32);
                    for_range(0u32.expr()..dim, |i| {
                        let sample = render_state.samples.var().read(offset + i).var();
                        *sample.cur = sample.backup;
                        *sample.last_modified = sample.modified_backup;
                        render_state.samples.var().write(offset + i, **sample);
                    });
                };
                if !is_large_step {
                    *state.n_mutations += 1;
                };
            }
            _ => todo!(),
        }
    }
    #[tracked]
    fn advance_chain(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        render_state: &RenderState,
        film: &Film,
        mutations_per_chain: Expr<u32>,
        contribution: Expr<f32>,
    ) {
        let i = dispatch_id().x;
        let markov_states = render_state.states.var();
        let sampler = IndependentSampler::from_pcg32(render_state.rng_states.var().read(i).var());
        let state = markov_states.read(i).var();
        let (cur_color, cur_swl) = render_state.cur_colors.read(i);
        let cur_color_v = ColorVar::new(cur_color);
        let cur_swl_v = cur_swl.var();
        for_range(0u32.expr()..mutations_per_chain, |_| {
            // we are about to overflow
            if state.cur_iter.eq(u32::MAX - 1) {
                // cpu_dbg!(i);
                let dim = self.sample_dimension();
                for_range(0u32.expr()..(dim as u32).expr(), |j| {
                    let sample = render_state.samples.var().read(i * dim as u32 + j).var();
                    if sample.last_modified.lt(state.last_large_iter) {
                        *sample.cur = sampler.next_1d();
                    };
                    *sample.last_modified = 0u32;
                });
                *state.cur_iter -= state.last_large_iter;
                *state.last_large_iter = 0u32;
            };
            self.mutate_chain(
                scene,
                color_pipeline,
                render_state,
                film,
                contribution,
                state,
                cur_color_v,
                cur_swl_v,
                &sampler,
            );
        });

        render_state
            .cur_colors
            .write(i, cur_color_v.load(), **cur_swl_v);
        render_state.rng_states.var().write(i, sampler.state.load());
        markov_states.write(i, state.load());
    }

    fn render_loop(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        state: &RenderState,
        film: &mut Film,
        session: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        let npixels = resolution.x * resolution.y;

        let kernel = self.device.create_kernel::<fn(u32, f32)>(
            &|mutations_per_chain: Expr<u32>, contribution: Expr<f32>| {
                if is_cpu_backend() {
                    let num_threads = std::thread::available_parallelism().unwrap().get();
                    if self.n_chains <= num_threads * 20 {
                        set_block_size([1, 1, 1]);
                    } else {
                        set_block_size([(num_threads / 20).clamp(1, 256) as u32, 1, 1]);
                    }
                } else {
                    set_block_size([256, 1, 1]);
                }
                self.advance_chain(
                    scene,
                    color_pipeline,
                    state,
                    film,
                    mutations_per_chain,
                    contribution,
                )
            },
        );
        log::info!(
            "Render kernel has {} arguments, {} captures!",
            kernel.num_arguments(),
            kernel.num_capture_arguments()
        );
        let reconstruct = |film: &mut Film, spp: u32| {
            let states = state.states.copy_to_vec();
            let mut b = state.b_init as f64;
            let mut b_cnt = state.b_init_cnt as u64;
            let mut accepted = 0u64;
            let mut mutations = 0u64;
            for s in &states {
                b += s.b as f64;
                b_cnt += s.b_cnt as u64;
                accepted += s.n_accepted as u64;
                mutations += s.n_mutations as u64;
            }
            let accept_rate = accepted as f64 / mutations as f64;
            let b = b / b_cnt as f64;
            log::info!("#indenpentent proposals: {}", b_cnt);
            log::info!("Normalization factor: {}", b);
            log::info!("Acceptance rate: {:.2}%", accept_rate * 100.0);
            film.set_splat_scale(b as f32 / spp as f32);
            if let Some(channel) = &session.display {
                film.copy_to_rgba_image(channel.screen_tex(), false);
                channel.notify_update();
            }
        };
        let mut acc_time = 0.0f64;
        let mut stats = RenderStats::default();
        {
            let mut cnt = 0;
            let spp_per_pass = self.pt.spp_per_pass;
            let contribution = {
                let n_mutations = npixels as u64 * self.pt.spp as u64;
                let mutations_per_chain = (n_mutations / self.n_chains as u64).max(1);
                n_mutations as f64 / (mutations_per_chain as f64 * self.n_chains as f64)
            } as f32;
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
                kernel.dispatch(
                    [self.n_chains as u32, 1, 1],
                    &mutations_per_chain,
                    &contribution,
                );
                progress.inc(cur_pass as u64);
                cnt += cur_pass;
                let toc = Instant::now();
                acc_time += toc.duration_since(tic).as_secs_f64();
                if session.save_intermediate || session.display.is_some() {
                    reconstruct(film, cnt);
                }
                if session.save_intermediate {
                    let output_image: Tex2d<Float4> = self.device.create_tex2d(
                        PixelStorage::Float4,
                        scene.camera.resolution().x,
                        scene.camera.resolution().y,
                        1,
                    );
                    reconstruct(film, cnt);
                    film.copy_to_rgba_image(&output_image, true);
                    let path = format!("{}-{}.exr", session.name, cnt);
                    util::write_image(&output_image, &path);
                    stats.intermediate.push(IntermediateStats {
                        time: acc_time,
                        spp: cnt,
                        path,
                    });
                }
            }
            progress.finish();
            if session.save_stats {
                let file = File::create(format!("{}.json", session.name)).unwrap();
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
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler_config: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        options: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig: {:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );

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
                    max_depth: 1,
                    rr_depth: 1,
                    spp: self.config.direct_spp as u32,
                    indirect_only: false,
                    spp_per_pass: self.pt.spp_per_pass,
                    use_nee: self.pt.use_nee,
                    ..Default::default()
                },
            );
            direct.render(
                scene.clone(),
                sampler_config,
                color_pipeline,
                film,
                &Default::default(),
            );
        }
        let render_state = self.bootstrap(&scene, film.filter(), color_pipeline);
        self.render_loop(&scene, color_pipeline, &render_state, film, options);
    }
}

pub fn render(
    device: Device,
    scene: Arc<Scene>,
    sampler: SamplerConfig,
    color_pipeline: ColorPipeline,
    film: &mut Film,
    config: &Config,
    options: &RenderSession,
) {
    let mcmc = McmcOpt::new(device.clone(), config.clone());
    mcmc.render(scene, sampler, color_pipeline, film, options);
}
