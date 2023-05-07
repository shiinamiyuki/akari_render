use rand::{thread_rng, Rng};

use super::pt::PathTracer;
use super::Integrator;
use crate::{
    color::*,
    film::*,
    geometry::*,
    interaction::*,
    sampler::{
        mcmc::{IsotropicGaussianMutation, LargeStepMutation, Mutation},
        *,
    },
    scene::*,
    surface::Bsdf,
    util::alias_table::AliasTable,
    *,
};

#[derive(Clone, Copy, Debug)]
pub enum Method {
    Kelemen {
        small_sigma: f32,
        large_step_prob: f32,
    },
    LangevinOnline,
    LangevinHybrid,
}
pub struct MCMC {
    pub device: Device,
    pub pt: PathTracer,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct MarkovState {
    chain_id: u32,
    cur_color: FlatColor,
    cur_pixel: Uint2,
    cur_f: f32,
    b: f32,
    b_cnt: u64,
    n_accepted: u64,
    n_mutations: u64,
}
struct RenderState {
    rng_states: Buffer<u32>,
    samples: Buffer<f32>,
    states: Buffer<MarkovState>,
    b_init: f32,
    b_init_cnt: u32,
}
impl MCMC {
    fn sample_dimension(&self) -> usize {
        self.pt.max_depth as usize * 8
    }
    pub fn new(
        device: Device,
        spp: u32,
        spp_per_pass: u32,
        max_depth: u32,
        method: Method,
        n_chains: usize,
        n_bootstrap: usize,
    ) -> Self {
        Self {
            device: device.clone(),
            pt: PathTracer::new(device.clone(), spp, spp_per_pass, max_depth),
            method,
            n_chains,
            n_bootstrap,
        }
    }
    fn scalar_contribution(&self, color: &Color) -> Expr<f32> {
        match color {
            Color::Rgb(v) => v.reduce_max(),
            Color::Spectral(_) => todo!(),
        }
    }
    fn evaluate<S: IndependentSampler + Clone>(
        &self,
        scene: &Scene,
        color_repr: &ColorRepr,
        independent: S,
        sample: PrimarySample,
    ) -> (Expr<Uint2>, Color, Expr<f32>) {
        let sampler = IndependentReplaySampler::new(independent, sample);
        sampler.start();
        let res = const_(scene.camera.resolution());
        let p = sampler.next_2d() * res.float();
        let p = p.int().clamp(0, res.int() - 1);
        let (ray, ray_color, ray_w) = scene.camera.generate_ray(p.uint(), &sampler, color_repr);
        let l = self.pt.radiance(scene, ray, &sampler, &color_repr) * ray_color * ray_w;
        (p.uint(), l.clone(), self.scalar_contribution(&l))
    }
    fn bootstrap(&self, scene: &Scene, color_repr: &ColorRepr) -> luisa::Result<RenderState> {
        let mut rng = thread_rng();
        let seeds = self
            .device
            .create_buffer_from_fn(self.n_bootstrap + self.n_chains, |_| rng.gen::<u32>())?;

        let fs = self
            .device
            .create_buffer_from_fn(self.n_bootstrap, |_| 0.0f32)?;
        self.device
            .create_kernel::<()>(&|| {
                let i = dispatch_id().x();
                let seed = seeds.var().read(i);
                let lcg_sampler = LcgSampler {
                    state: var!(u32, seed),
                };
                let sample = VLArrayVar::<f32>::zero(self.sample_dimension());
                for_range(const_(0)..sample.len().int(), |i| {
                    let i = i.uint();
                    sample.write(i, lcg_sampler.next_1d());
                });
                let sample = PrimarySample { values: sample };
                let (_p, _l, f) = self.evaluate(scene, color_repr, lcg_sampler.clone(), sample);
                fs.var().write(i, f);
            })?
            .dispatch([self.n_bootstrap as u32, 1, 1])?;

        let weights = fs.copy_to_vec();
        let b = weights.iter().sum::<f32>();
        let at = AliasTable::new(self.device.clone(), &weights)?;
        let states = self.device.create_buffer(self.n_chains)?;
        let sample_buffer = self
            .device
            .create_buffer(self.sample_dimension() * self.n_chains)?;
        self.device
            .create_kernel::<()>(&|| {
                let i = dispatch_id().x();
                let seed = seeds.var().read(i + self.n_bootstrap as u32);
                let lcg_sampler = LcgSampler {
                    state: var!(u32, seed),
                };
                let (seed_idx, _) = at.sample(lcg_sampler.next_2d());
                let seed = seeds.var().read(seed_idx);
                let lcg_sampler = LcgSampler {
                    state: var!(u32, seed),
                };
                let sample = VLArrayVar::<f32>::zero(self.sample_dimension());
                for_range(const_(0)..sample.len().int(), |i| {
                    let i = i.uint();
                    sample.write(i, lcg_sampler.next_1d());
                });

                for_range(const_(0)..sample.len().int(), |j| {
                    let dim = self.sample_dimension();
                    let j = j.uint();
                    sample_buffer
                        .var()
                        .write(i * dim as u32 + j, sample.read(j));
                });
                let sample = PrimarySample { values: sample };
                let (p, l, f) = self.evaluate(scene, color_repr, lcg_sampler.clone(), sample);
                let l = l.flatten();
                let state = MarkovStateExpr::new(i, l, p, f, f, 1, 0, 0);
                states.var().write(i, state);
            })?
            .dispatch([self.n_chains as u32, 1, 1])?;
        let rng_states = self
            .device
            .create_buffer_from_fn(self.n_chains, |_| rng.gen::<u32>())?;
        Ok(RenderState {
            rng_states,
            samples: sample_buffer,
            states,
            b_init: b,
            b_init_cnt: self.n_bootstrap as u32,
        })
    }
    fn mutate_chain<S: IndependentSampler + Clone>(
        &self,
        scene: &Scene,
        color_repr: &ColorRepr,
        _render_state: &RenderState,
        film: &Film,
        contribution: Expr<f32>,
        state: Var<MarkovState>,
        sample: PrimarySample,
        rng: S,
    ) {
        // select a mutation strategy
        match self.method {
            Method::Kelemen {
                small_sigma,
                large_step_prob,
            } => {
                let is_large_step = rng.next_1d().cmplt(large_step_prob);
                let proposal = if_!(is_large_step, {
                    let large = LargeStepMutation{};
                    large.mutate(&sample, &rng)
                }, else {
                    let small = IsotropicGaussianMutation{sigma: small_sigma.into()};
                    small.mutate(&sample, &rng)
                });
                let clamped = proposal.sample.clamped();
                let (proposal_p, proposal_color, f) =
                    self.evaluate(scene, color_repr, rng.clone(), clamped);
                let proposal_f = f;
                if_!(is_large_step, {
                    state.set_b(state.b().load() + proposal_f);
                    state.set_b_cnt(state.b_cnt().load() + 1);
                });
                let cur_f = state.cur_f().load();
                let cur_p = state.cur_pixel().load();
                let cur_color = Color::from_flat(state.cur_color().load(), color_repr);
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
                    state.set_cur_color(proposal_color.flatten());
                    state.set_cur_pixel(proposal_p);
                    state.set_n_accepted(state.n_accepted().load() + 1);
                    sample.values.store(clamped.values.load());
                });
                state.set_n_mutations(state.n_mutations().load() + 1);
            }
            _ => todo!(),
        }
    }
    fn advance_chain(
        &self,
        scene: &Scene,
        color_repr: &ColorRepr,
        render_state: &RenderState,
        film: &Film,
        mutations_per_chain: Expr<u32>,
        contribution: Expr<f32>,
    ) {
        let i = dispatch_id().x();
        let markov_states = render_state.states.var();
        let sampler = LcgSampler {
            state: var!(u32, render_state.rng_states.var().read(i)),
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
        for_range(const_(0)..mutations_per_chain.int(), |_| {
            self.mutate_chain(
                scene,
                color_repr,
                render_state,
                film,
                contribution,
                state,
                sample,
                sampler.clone(),
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
        render_state.rng_states.var().write(i, sampler.state.load());
        markov_states.write(i, state.load());
    }

    fn render_loop(
        &self,
        scene: &Scene,
        color_repr: &ColorRepr,
        state: &RenderState,
        film: &mut Film,
    ) -> luisa::Result<()> {
        let resolution = scene.camera.resolution();
        let npixels = resolution.x * resolution.y;

        let kernel = self.device.create_kernel::<(u32, f32)>(
            &|mutations_per_chain: Expr<u32>, contribution: Expr<f32>| {
                set_block_size([1, 1, 1]);
                self.advance_chain(
                    scene,
                    color_repr,
                    state,
                    film,
                    mutations_per_chain,
                    contribution,
                )
            },
        )?;
        {
            let mut cnt = 0;
            let spp_per_pass = self.pt.spp_per_pass;
            let progress = util::create_progess_bar(self.pt.spp as usize, "spp");
            while cnt < self.pt.spp {
                let cur_pass = (self.pt.spp - cnt).min(spp_per_pass);
                let n_mutations = npixels as u64 * cur_pass as u64;
                let mutations_per_chain = n_mutations / self.n_chains as u64;
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
                )?;
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
        log::info!("Normalization factor: {}", b);
        log::info!("Acceptance rate: {:.2}%", accept_rate * 100.0);
        film.set_splat_scale(b as f32 / self.pt.spp as f32);
        Ok(())
    }
}
impl Integrator for MCMC {
    fn render(&self, scene: &Scene, film: &mut Film) -> luisa::Result<()> {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}, spp: {}",
            resolution.x,
            resolution.y,
            self.pt.spp
        );

        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        film.clear();
        let color_repr = ColorRepr::Rgb;
        let render_state = self.bootstrap(scene, &color_repr)?;
        self.render_loop(scene, &color_repr, &render_state, film)?;
        Ok(())
    }
}
