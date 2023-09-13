use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;

use super::mcmc::{Config, Method};
use super::pt::{self, PathTracer, PathTracerBase};
use super::{Integrator, IntermediateStats, RenderOptions, RenderStats};
use crate::sampler::mcmc::{
    IsotropicExponentialMutation, KelemenMutationRecord, KelemenMutationRecordExpr, KELEMEN_MUTATE,
};
use crate::sampling::sample_gaussian;
use crate::util::distribution::resample_with_f64;
use crate::{
    color::*,
    film::*,
    sampler::{
        mcmc::{IsotropicGaussianMutation, LargeStepMutation, Mutation},
        *,
    },
    scene::*,
    *,
};

pub struct SinglePathMcmc {
    pub device: Device,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    max_depth: u32,
    min_depth: u32,
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
    depth: u32,
}
struct RenderState {
    rng_states: Buffer<Pcg32>,
    samples: Buffer<f32>,
    states: Buffer<MarkovState>,
    cur_colors: ColorBuffer,
    b_init: f64,
    b_init_cnt: usize,
}
pub struct McmcSampler<'a, F> {
    pub base: &'a IndependentSampler,
    pub samples: &'a Buffer<f32>,
    pub get_index: F,
    pub cur_dim: Var<u32>,
    pub mcmc_dim: Expr<u32>,
}
impl<'a, F> McmcSampler<'a, F> {
    pub fn new(
        base: &'a IndependentSampler,
        samples: &'a Buffer<f32>,
        get_index: F,
        mcmc_dim: Expr<u32>,
    ) -> Self
    where
        F: Fn(Expr<u32>) -> Expr<u32>,
    {
        Self {
            base,
            samples,
            get_index,
            cur_dim: var!(u32, 0),
            mcmc_dim,
        }
    }
}

impl<'a, F> Sampler for McmcSampler<'a, F>
where
    F: Fn(Expr<u32>) -> Expr<u32>,
{
    fn next_1d(&self) -> Float {
        if_!(
            self.cur_dim.load().cmplt(self.mcmc_dim),
            {
                let ret = self.samples.read((self.get_index)(*self.cur_dim));
                self.cur_dim.store(self.cur_dim.load() + 1);
                ret
            },
            else,
            {
                lc_unreachable!();
                const_(0.0f32)
            }
        )
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
    // pub is_image_mutation: Expr<bool>,
}
impl Mutator {
    pub fn mutate(
        &self,
        samples: &Buffer<f32>,
        start_dim: Expr<u32>,
        end_dim: Expr<u32>,
        get_index: &impl Fn(Expr<u32>) -> Expr<u32>,
        get_back_index: &impl Fn(Expr<u32>) -> Expr<u32>,
        rng: &IndependentSampler,
    ) {
        for_range(start_dim..end_dim, |i| {
            self.mutate_one(samples, i, get_index, get_back_index, rng);
        });
    }
    pub fn reject(
        &self,
        samples: &Buffer<f32>,
        start_dim: Expr<u32>,
        end_dim: Expr<u32>,
        get_index: &impl Fn(Expr<u32>) -> Expr<u32>,
        get_back_index: &impl Fn(Expr<u32>) -> Expr<u32>,
    ) {
        for_range(start_dim..end_dim, |i| {
            let sample_idx = get_index(i);
            let backup_idx = get_back_index(i);
            let x = samples.read(backup_idx);
            samples.write(sample_idx, x);
        });
    }
    pub fn mutate_one(
        &self,
        samples: &Buffer<f32>,
        i: Expr<u32>, // dim
        get_index: &impl Fn(Expr<u32>) -> Expr<u32>,
        get_back_index: &impl Fn(Expr<u32>) -> Expr<u32>,
        rng: &IndependentSampler,
    ) {
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
                let u = rng.next_1d();
                let sample_idx = get_index(i);
                let backup_idx = get_back_index(i);
                let x = samples.read(sample_idx);
                samples.write(backup_idx, x);
                if_!(
                    self.is_large_step,
                    {
                        samples.write(sample_idx, u);
                    },
                    else,
                    {
                        if exponential_mutation {
                            let record = var!(
                                KelemenMutationRecord,
                                KelemenMutationRecordExpr::new(
                                    x,
                                    u,
                                    kelemen_mutate_size_low,
                                    kelemen_mutate_size_high,
                                    kelemen_log_ratio,
                                    0.0
                                )
                            );
                            KELEMEN_MUTATE.call(record);
                            samples.write(sample_idx, *record.mutated());
                        } else {
                            let dv = sample_gaussian(u);
                            let new = x + dv * small_sigma;
                            let new = new - new.floor();
                            // lc_assert!(new.is_finite());
                            let new = select(new.is_finite(), new, const_(0.0f32));
                            samples.write(sample_idx, new);
                        }
                    }
                );
            }
        }
    }
}
impl SinglePathMcmc {
    fn sample_dimension(&self, depth: u32) -> u32 {
        4 + (1 + depth) * 3 + 3
    }
    fn sample_dimension_device(&self, depth: Expr<u32>) -> Expr<u32> {
        4 + (1 + depth) * 3 + 3
    }
    pub fn new(device: Device, config: Config) -> Self {
        // assert_eq!(
        //     config.n_chains % Self::SAMPLE_BUFFER_AOSOA_SIZE as usize,
        //     0,
        //     "n_chains must be a multiple of {}",
        //     Self::SAMPLE_BUFFER_AOSOA_SIZE
        // );
        Self {
            device: device.clone(),
            method: config.method,
            n_chains: config.n_chains,
            n_bootstrap: config.n_bootstrap,
            max_depth: config.max_depth,
            min_depth: if config.direct_spp >= 0 { 2 } else { 0 },
            config,
        }
    }
    pub fn scalar_contribution(color: &Color) -> Expr<f32> {
        color.max().clamp(0.0, 1e5)
        // const_(1.0f32)
    }
    fn evaluate(
        &self,
        scene: &Arc<Scene>,
        filter: PixelFilter,
        eval: &Evaluators,
        independent: &IndependentSampler,
        samples: &Buffer<f32>,
        get_index: impl Fn(Expr<u32>) -> Expr<u32>,
        depth: Expr<u32>,
    ) -> (Expr<Uint2>, Color, Expr<SampledWavelengths>, Expr<f32>) {
        let sampler = McmcSampler::new(
            independent,
            samples,
            get_index,
            self.sample_dimension_device(depth),
        );
        sampler.start();
        let res = const_(scene.camera.resolution());
        let p = sampler.next_2d() * res.float();
        let p = p.int().clamp(0, res.int() - 1);
        let swl = def(sample_wavelengths(eval.color_repr(), &sampler));
        let (ray, ray_color, ray_w) =
            scene
                .camera
                .generate_ray(filter, p.uint(), &sampler, eval.color_repr(), *swl);
        let l = {
            let w = ray_color * ray_w;
            let pt = PathTracerBase::new(
                scene,
                eval,
                depth,
                depth,
                self.config.use_nee,
                self.config.direct_spp >= 0,
                swl,
            );
            pt.run_at_depth(ray, depth, &sampler);
            pt.radiance.load() * w
        };
        (p.uint(), l, *swl, Self::scalar_contribution(&l))
    }
    const SAMPLE_BUFFER_AOSOA_SIZE: u32 = 8;

    // layout:
    // [[f32 x 8] x 2] x dims] x (n_chains / 8)
    fn _sample_index_aosoa(
        &self,
        chain_id: Expr<u32>,
        dim: Expr<u32>,
        is_backup: Expr<u32>,
    ) -> Expr<u32> {
        // (chain_id / Self::SAMPLE_BUFFER_AOSOA_SIZE)
        //     * Self::SAMPLE_BUFFER_AOSOA_SIZE
        //     * 2
        //     * self.sample_dimension(self.config.max_depth)
        //     + dim * Self::SAMPLE_BUFFER_AOSOA_SIZE * 2
        //     + (chain_id % Self::SAMPLE_BUFFER_AOSOA_SIZE)
        //     + is_backup * Self::SAMPLE_BUFFER_AOSOA_SIZE
        chain_id * self.sample_dimension(self.max_depth) * 2 + dim * 2 + is_backup
    }
    fn sample_index_aosoa(&self, chain_id: Expr<u32>, dim: Expr<u32>) -> Expr<u32> {
        self._sample_index_aosoa(chain_id, dim, const_(0u32))
    }
    fn backup_index_aosoa(&self, chain_id: Expr<u32>, dim: Expr<u32>) -> Expr<u32> {
        self._sample_index_aosoa(chain_id, dim, const_(1u32))
    }
    fn normalization_factor_correction(&self) -> f32 {
        (self.max_depth - self.min_depth + 1) as f32
    }
    fn sample_depth(&self, sampler: &dyn Sampler) -> Expr<u32> {
        let range = self.max_depth + 1 - self.min_depth;
        (self.min_depth + (sampler.next_1d() * range as f32).uint())
            .clamp(self.min_depth, self.config.max_depth as i32)
    }
    fn bootstrap(&self, scene: &Arc<Scene>, filter: PixelFilter, eval: &Evaluators) -> RenderState {
        let seeds = init_pcg32_buffer(self.device.clone(), self.n_bootstrap);
        let fs = self
            .device
            .create_buffer_from_fn(self.n_bootstrap, |_| 0.0f32);
        let sample_buffer = self.device.create_buffer(
            self.sample_dimension(self.config.max_depth) as usize * self.n_bootstrap,
        );
        let bootstrap_kernel = self.device.create_kernel::<fn()>(&|| {
            let i = dispatch_id().x();
            let seed = seeds.var().read(i);
            let sampler = IndependentSampler::from_pcg32(var!(Pcg32, seed));

            let depth = self.sample_depth(&sampler);
            // cpu_dbg!(depth);
            for_range(const_(0u32)..self.sample_dimension_device(depth), |d| {
                sample_buffer.write(
                    self.sample_dimension(self.config.max_depth) * i + d,
                    sampler.next_1d(),
                );
            });
            let (_p, _l, _swl, f) = self.evaluate(
                scene,
                filter,
                eval,
                &sampler,
                &sample_buffer,
                |d| self.sample_dimension(self.config.max_depth) * i + d,
                depth,
            );
            fs.var().write(i, f);
        });
        let t = Instant::now();
        bootstrap_kernel.dispatch([self.n_bootstrap as u32, 1, 1]);

        let weights = fs.copy_to_vec();
        let (b, resampled) = resample_with_f64(&weights, self.n_chains);
        log::info!("Bootstrap finished in {:.2}s", t.elapsed().as_secs_f64());
        assert!(b > 0.0, "Bootstrap failed, please retry with more samples");
        log::info!(
            "Normalization factor initial estimate: {}",
            b / self.n_bootstrap as f64 * self.normalization_factor_correction() as f64
        );
        let resampled = self.device.create_buffer_from_slice(&resampled);
        let states = self.device.create_buffer(self.n_chains);
        let cur_colors = ColorBuffer::new(self.device.clone(), self.n_chains, eval.color_repr());
        let sample_buffer = self.device.create_buffer(
            self.sample_dimension(self.max_depth) as usize * 2 * ((self.n_chains + 7) / 8) * 8,
        );
        self.device
            .create_kernel::<fn()>(&|| {
                let i = dispatch_id().x();
                let seed_idx = resampled.var().read(i);
                let seed = seeds.var().read(seed_idx);
                let sampler = IndependentSampler::from_pcg32(var!(Pcg32, seed));

                let depth = self.sample_depth(&sampler);
                for_range(const_(0u32)..self.sample_dimension_device(depth), |d| {
                    sample_buffer.write(self.sample_index_aosoa(i, d), sampler.next_1d());
                });
                let (p, l, swl, f) = self.evaluate(
                    scene,
                    filter,
                    eval,
                    &sampler,
                    &sample_buffer,
                    |d| self.sample_index_aosoa(i, d),
                    depth,
                );

                // cpu_dbg!(make_float2(f, fs.var().read(seed_idx)));
                let sigma = match &self.method {
                    Method::Kelemen { small_sigma, .. } => *small_sigma,
                    _ => todo!(),
                };
                cur_colors.write(i, l, swl);
                let state = MarkovStateExpr::new(i, p, f, 0.0, 0, 0, 0, sigma, depth);
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
            b_init_cnt: self.n_bootstrap,
        }
    }
    fn mutate_chain(
        &self,
        scene: &Arc<Scene>,
        eval: &Evaluators,
        film: &Film,
        contribution: Expr<f32>,
        state: Var<MarkovState>,
        cur_color_v: ColorVar,
        cur_swl: Var<SampledWavelengths>,
        samples: &Buffer<f32>,
        rng: &IndependentSampler,
    ) {
        let res = const_(scene.camera.resolution()).float();
        // select a mutation strategy
        match self.method {
            Method::Kelemen {
                small_sigma: _,
                large_step_prob,
                adaptive,
                exponential_mutation,
                image_mutation_size,
                image_mutation_prob,
            } => {
                let chain_id = *state.chain_id();
                let is_large_step = rng.next_1d().cmplt(large_step_prob);
                let new_depth = var!(u32);
                if_!(
                    is_large_step,
                    {
                        new_depth.store(self.sample_depth(rng));
                    },
                    else,
                    {
                        new_depth.store(*state.depth());
                    }
                );
                let mutator = Mutator {
                    is_large_step,
                    method: self.method,
                };
                // perform mutation
                let mutation_end_dim = self.sample_dimension_device(*new_depth);
                mutator.mutate(
                    samples,
                    const_(0u32),
                    mutation_end_dim,
                    &|dim| self.sample_index_aosoa(chain_id, dim),
                    &|dim| self.backup_index_aosoa(chain_id, dim),
                    rng,
                );

                let (proposal_p, proposal_color, proposal_swl, f) = self.evaluate(
                    scene,
                    film.filter(),
                    eval,
                    &rng,
                    samples,
                    |dim| self.sample_index_aosoa(chain_id, dim),
                    *new_depth,
                );
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
                    proposal_swl,
                    accept * contribution,
                );
                film.add_splat(
                    cur_p.float(),
                    &(cur_color / cur_f),
                    *cur_swl,
                    (1.0 - accept) * contribution,
                );
                if_!(rng.next_1d().cmplt(accept) & state.b_cnt().cmplt(1024 * 1024), {
                    state.set_cur_f(proposal_f);
                    cur_color_v.store(proposal_color);
                    cur_swl.store(proposal_swl);
                    state.set_cur_pixel(proposal_p);
                    if_!(!is_large_step, {
                        state.set_n_accepted(state.n_accepted().load() + 1);
                    });
                    if_!(is_large_step, {
                        *state.depth().get_mut() = *new_depth;
                    });
                }, else {
                    mutator.reject(
                        samples,
                        const_(0u32),
                        mutation_end_dim,
                        &|dim| self.sample_index_aosoa(chain_id, dim),
                        &|dim| self.backup_index_aosoa(chain_id, dim),
                    );

                });
                if_!(!is_large_step, {
                    state.set_n_mutations(state.n_mutations().load() + 1);
                    if adaptive {
                        let r =
                            state.n_accepted().load().float() / state.n_mutations().load().float();
                        const OPTIMAL_ACCEPT_RATE: f32 = 0.234;
                        if_!(state.n_mutations().load().cmpgt(50), {
                            let new_sigma = state.sigma().load()
                                + (r - OPTIMAL_ACCEPT_RATE) / state.n_mutations().load().float();
                            let new_sigma = new_sigma.clamp(1e-5, 0.1);
                            if_!(state.chain_id().load().cmpeq(0), {
                                cpu_dbg!(make_float3(r, state.sigma().load(), new_sigma));
                            });
                            state.sigma().store(new_sigma);
                        });
                    }
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
        let sampler =
            IndependentSampler::from_pcg32(var!(Pcg32, render_state.rng_states.var().read(i)));
        let state = var!(MarkovState, markov_states.read(i));
        let (cur_color, cur_swl) = render_state.cur_colors.read(i);
        let cur_color_v = ColorVar::new(cur_color);
        let cur_swl_v = def(cur_swl);
        for_range(const_(0)..mutations_per_chain.int(), |_| {
            self.mutate_chain(
                scene,
                eval,
                film,
                contribution,
                state,
                cur_color_v,
                cur_swl_v,
                &render_state.samples,
                &sampler,
            );
        });
        render_state
            .cur_colors
            .write(i, cur_color_v.load(), *cur_swl_v);
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
            let b = b / b_cnt as f64 * self.normalization_factor_correction() as f64;
            log::info!("#indenpentent proposals: {}", b_cnt);
            log::info!("Normalization factor: {}", b);
            log::info!("Acceptance rate: {:.2}%", accept_rate * 100.0);
            film.set_splat_scale(b as f32 / spp as f32);
        };
        let mut acc_time = 0.0f64;
        let mut stats = RenderStats::default();
        let spp_per_pass = self.config.spp_per_pass;
        let spp = self.config.spp;
        {
            let mut cnt = 0;
            let progress = util::create_progess_bar(spp as usize, "spp");
            // since mutations_per_chain is truncated, we need to compensate for the truncation error
            // mutations_per_chain * n_chains * contribution = n_mutations
            let contribution = {
                let n_mutations = npixels as u64 * spp as u64;
                let mutations_per_chain = (n_mutations / self.n_chains as u64).max(1);
                n_mutations as f64 / (mutations_per_chain as f64 * self.n_chains as f64)
            } as f32;
            while cnt < spp {
                let tic = Instant::now();
                let cur_pass = (spp - cnt).min(spp_per_pass);
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
        reconstruct(film, spp);
    }
}
impl Integrator for SinglePathMcmc {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler_config: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        options: &RenderOptions,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig: {:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        let evaluators = scene.evaluators(color_pipeline, None);
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
                    spp_per_pass: self.config.spp_per_pass,
                    use_nee: self.config.use_nee,
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
        let render_state = self.bootstrap(&scene, film.filter(), &evaluators);
        self.render_loop(&scene, &evaluators, &render_state, film, options);
    }
}

pub fn render(
    device: Device,
    scene: Arc<Scene>,
    sampler: SamplerConfig,
    color_pipeline: ColorPipeline,
    film: &mut Film,
    config: &Config,
    options: &RenderOptions,
) {
    let mcmc = SinglePathMcmc::new(device.clone(), config.clone());
    mcmc.render(scene, sampler, color_pipeline, film, options);
}
