use std::sync::Arc;

use super::pt::{self, PathTracer};
use super::Integrator;
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
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Method {
    #[serde(rename = "kelemen")]
    Kelemen {
        small_sigma: f32,
        large_step_prob: f32,
        adaptive: bool,
    },
    LangevinOnline,
    LangevinHybrid,
}
impl Default for Method {
    fn default() -> Self {
        Method::Kelemen {
            small_sigma: 0.01,
            large_step_prob: 0.1,
            adaptive: false,
        }
    }
}
pub struct Mcmc {
    pub device: Device,
    pub pt: PathTracer,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    config: Config,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub rr_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
    pub direct_spp: i32,
}
impl Default for Config {
    fn default() -> Self {
        let default_pt = pt::Config::default();
        Self {
            spp: default_pt.spp,
            spp_per_pass: default_pt.spp_per_pass,
            use_nee: default_pt.use_nee,
            max_depth: default_pt.max_depth,
            rr_depth: default_pt.rr_depth,
            method: Method::default(),
            n_chains: 512,
            n_bootstrap: 100000,
            direct_spp: 64,
        }
    }
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
    samples: Buffer<f32>,
    states: Buffer<MarkovState>,
    cur_colors: ColorBuffer,
    b_init: f32,
    b_init_cnt: u32,
}
impl Mcmc {
    fn sample_dimension(&self) -> usize {
        4 + self.pt.max_depth as usize * (3 + 3 + 1)
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
            config,
        }
    }
    fn scalar_contribution(&self, color: &Color) -> Expr<f32> {
        color.max().clamp(0.0, 1e5)
    }
    fn evaluate(
        &self,
        scene: &Arc<Scene>,
        eval: &Evaluators,
        independent: &IndependentSampler,
        sample: PrimarySample,
    ) -> (Expr<Uint2>, Color, Expr<f32>) {
        let sampler = IndependentReplaySampler::new(independent, sample);
        sampler.start();
        let res = const_(scene.camera.resolution());
        let p = sampler.next_2d() * res.float();
        let p = p.int().clamp(0, res.int() - 1);
        let (ray, ray_color, ray_w) =
            scene
                .camera
                .generate_ray(p.uint(), &sampler, eval.color_repr);
        let l = self.pt.radiance(scene, ray, &sampler, eval) * ray_color * ray_w;
        (p.uint(), l.clone(), self.scalar_contribution(&l))
    }
    fn bootstrap(&self, scene: &Arc<Scene>, eval: &Evaluators) -> RenderState {
        let seeds = init_pcg32_buffer(self.device.clone(), self.n_bootstrap + self.n_chains);

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
                let sample = VLArrayVar::<f32>::zero(self.sample_dimension());
                for_range(const_(0)..sample.len().int(), |i| {
                    let i = i.uint();
                    sample.write(i, sampler.next_1d());
                });
                let sample = PrimarySample { values: sample };
                let (_p, _l, f) = self.evaluate(scene, eval, &sampler, sample);
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
                let sample = VLArrayVar::<f32>::zero(self.sample_dimension());
                for_range(const_(0)..sample.len().int(), |i| {
                    let i = i.uint();
                    sample.write(i, sampler.next_1d());
                });

                for_range(const_(0)..sample.len().int(), |j| {
                    let dim = self.sample_dimension();
                    let j = j.uint();
                    sample_buffer
                        .var()
                        .write(i * dim as u32 + j, sample.read(j));
                });
                let sample = PrimarySample { values: sample };
                let (p, l, f) = self.evaluate(scene, eval, &sampler, sample);
                let sigma = match &self.method {
                    Method::Kelemen { small_sigma, .. } => *small_sigma,
                    _ => todo!(),
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
        film: &Film,
        contribution: Expr<f32>,
        state: Var<MarkovState>,
        cur_color_v: ColorVar,
        sample: PrimarySample,
        rng: &IndependentSampler,
    ) {
        // select a mutation strategy
        match self.method {
            Method::Kelemen {
                small_sigma: _,
                large_step_prob,
                adaptive,
            } => {
                let is_large_step = rng.next_1d().cmplt(large_step_prob);
                let proposal = if_!(is_large_step, {
                    let large = LargeStepMutation{};
                    large.mutate(&sample, &rng)
                }, else {
                    let small = IsotropicGaussianMutation{sigma: state.sigma().load()};
                    small.mutate(&sample, &rng)
                });
                let clamped = proposal.sample.clamped();
                let (proposal_p, proposal_color, f) = self.evaluate(scene, eval, &rng, clamped);
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
                    &(proposal_color.clone() / proposal_f * accept * contribution),
                );
                film.add_splat(
                    cur_p.float(),
                    &(cur_color / cur_f * (1.0 - accept * contribution)),
                );
                if_!(rng.next_1d().cmplt(accept), {
                    state.set_cur_f(proposal_f);
                    cur_color_v.store(proposal_color);
                    state.set_cur_pixel(proposal_p);
                    if_!(!is_large_step, {
                        state.set_n_accepted(state.n_accepted().load() + 1);
                    });
                    sample.values.store(clamped.values.load());
                });
                if_!(!is_large_step, {
                    state.set_n_mutations(state.n_mutations().load() + 1);
                    if adaptive {
                        let r =
                            state.n_accepted().load().float() / state.n_mutations().load().float();
                        const OPTIMAL_ACCEPT_RATE: f32 = 0.234;
                        let new_sigma = state.sigma().load()
                            + (r - OPTIMAL_ACCEPT_RATE) / state.n_mutations().load().float();
                        let new_sigma = new_sigma.clamp(1e-5, 0.1);
                        state.sigma().store(new_sigma);
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
        let sampler = IndependentSampler {
            state: var!(Pcg32, render_state.rng_states.var().read(i)),
        };
        let state = var!(MarkovState, markov_states.read(i));
        let sample = {
            let dim = self.sample_dimension();
            let sample = VLArrayVar::<f32>::zero(dim);
            lc_assert!(i.cmpeq(state.chain_id().load()));
            for_range(const_(0)..const_(dim as i32), |j| {
                let j = j.uint();
                sample.write(j, render_state.samples.var().read(i * dim as u32 + j));
            });
            PrimarySample { values: sample }
        };
        let cur_color_v = ColorVar::new(render_state.cur_colors.read(i));
        for_range(const_(0)..mutations_per_chain.int(), |_| {
            self.mutate_chain(
                scene,
                eval,
                film,
                contribution,
                state,
                cur_color_v,
                sample,
                &sampler,
            );
        });
        {
            let dim = self.sample_dimension();
            for_range(const_(0)..const_(dim as i32), |j| {
                let j = j.uint();
                render_state
                    .samples
                    .var()
                    .write(i * dim as u32 + j, sample.values.read(j));
            });
        }
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
    ) {
        let resolution = scene.camera.resolution();
        let npixels = resolution.x * resolution.y;

        let kernel = self.device.create_kernel::<(u32, f32)>(
            &|mutations_per_chain: Expr<u32>, contribution: Expr<f32>| {
                set_block_size([1, 1, 1]);
                self.advance_chain(scene, eval, state, film, mutations_per_chain, contribution)
            },
        );
        {
            let mut cnt = 0;
            let spp_per_pass = self.pt.spp_per_pass;
            let progress = util::create_progess_bar(self.pt.spp as usize, "spp");
            while cnt < self.pt.spp {
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
            }
            progress.finish();
        }
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
        film.set_splat_scale(b as f32 / self.pt.spp as f32);
    }
}
impl Integrator for Mcmc {
    fn render(&self, scene: Arc<Scene>, film: &mut Film) {
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
            direct.render(scene.clone(), film);
        }
        let render_state = self.bootstrap(&scene, &evaluators);
        self.render_loop(&scene, &evaluators, &render_state, film);
    }
}

pub fn render(device: Device, scene: Arc<Scene>, film: &mut Film, config: &Config) {
    let mcmc = Mcmc::new(device.clone(), config.clone());
    mcmc.render(scene, film);
}
