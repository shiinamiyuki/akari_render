use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;

use luisa::rtx::offset_ray_origin;
use serde::{Deserialize, Serialize};

use super::pt::{self, PathTracer, PathTracerBase};
use super::{Integrator, IntermediateStats, RenderSession, RenderStats};
use crate::geometry::{face_forward, Ray, RayExpr};
use crate::sampler::mcmc::{KelemenMutationRecord, KELEMEN_MUTATE};
use crate::sampling::sample_gaussian;
use crate::util::distribution::resample_with_f64;
use crate::{color::*, film::*, sampler::*, scene::*, *};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Method {
    #[serde(rename = "kelemen")]
    Kelemen {
        exponential_mutation: bool,
        small_sigma: f32,
        large_step_prob: f32,
        image_mutation_prob: f32,
        image_mutation_size: Option<f32>,
        depth_change_prob: f32,
        adaptive: bool,
    },
}
impl Default for Method {
    fn default() -> Self {
        Method::Kelemen {
            exponential_mutation: true,
            small_sigma: 0.01,
            large_step_prob: 0.1,
            image_mutation_prob: 0.0,
            image_mutation_size: None,
            depth_change_prob: 0.0,
            adaptive: false,
        }
    }
}
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
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
            method: Method::default(),
            n_chains: 512,
            n_bootstrap: 100000,
            direct_spp: 64,
        }
    }
}
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
#[value_new]
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
            cur_dim: u32::var_zeroed(),
            mcmc_dim,
        }
    }
}

impl<'a, F> Sampler for McmcSampler<'a, F>
where
    F: Fn(Expr<u32>) -> Expr<u32>,
{
    #[tracked]
    fn next_1d(&self) -> Expr<f32> {
        if self.cur_dim.load().lt(self.mcmc_dim) {
            let ret = self.samples.read((self.get_index)(**self.cur_dim));
            *self.cur_dim += 1;
            ret
        } else {
            lc_unreachable!();
            0.0f32.expr()
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
    pub small_step_dim_change: Expr<bool>,
    pub old_dim: Expr<u32>,
    pub new_dim: Expr<u32>,
    // pub is_image_mutation: Expr<bool>,
}
impl Mutator {
    pub fn mutate(
        &self,
        samples: &Buffer<f32>,
        get_index: &impl Fn(Expr<u32>) -> Expr<u32>,
        get_back_index: &impl Fn(Expr<u32>) -> Expr<u32>,
        rng: &IndependentSampler,
    ) {
        for_range(0u32.expr()..self.new_dim, |i| {
            self.mutate_one(samples, i, get_index, get_back_index, rng);
        });
    }
    pub fn reject(
        &self,
        samples: &Buffer<f32>,
        get_index: &impl Fn(Expr<u32>) -> Expr<u32>,
        get_back_index: &impl Fn(Expr<u32>) -> Expr<u32>,
    ) {
        for_range(0u32.expr()..self.new_dim, |i| {
            let sample_idx = get_index(i);
            let backup_idx = get_back_index(i);
            let x = samples.read(backup_idx);
            samples.write(sample_idx, x);
        });
    }
    #[tracked]
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
                let x = samples.read(sample_idx).var();
                samples.write(backup_idx, x);
                let new = if self.is_large_step {
                    u
                } else {
                    if self.small_step_dim_change & i.gt(self.old_dim) {
                        *x = rng.next_1d();
                    }
                    escape!({
                        if exponential_mutation {
                            track!({
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
                                **record.mutated
                            })
                        } else {
                            track!({
                                let dv = sample_gaussian(u);
                                let new = x + dv * small_sigma;
                                let new = new - new.floor();
                                // lc_assert!(new.is_finite());
                                let new = select(new.is_finite(), new, 0.0f32.expr());
                                new
                            })
                        }
                    })
                };
                samples.write(sample_idx, new);
            }
        }
    }
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct PathVertex {
    pub inst_id: u32,
    pub prim_id: u32,
    pub bary: Float2,
    pub beta: FlatColor,
    pub radiance: FlatColor,
    pub ray: Ray,
}

impl<'a> PathTracerBase<'a> {
    /// Sample a path prefix with length `target_depth - 1`
    /// and two final vertices with NEE and BSDF sampling respectively.
    #[tracked]
    pub fn run_at_depth(
        &self,
        ray: Expr<Ray>,
        target_depth: Expr<u32>,
        sampler: &dyn Sampler,
        vertices: Option<VLArrayVar<PathVertex>>,
    ) {
        let ray = ray.var();
        let u_light = sampler.next_3d();
        loop {
            let hit = self.next_intersection(**ray);
            if !hit {
                if let Some(vertices) = vertices {
                    // vertices.write(
                    //     *self.depth,
                    //     PathVertexExpr::new(u32::MAX, u32::MAX, Float2::expr(0.0, 0.0)),
                    // );
                    todo!()
                }
                let (direct, w) = self.hit_envmap(**ray);
                if self.depth.eq(target_depth) {
                    self.add_radiance(direct * w);
                };
                break_();
            }
            if let Some(vertices) = vertices {
                // vertices.write(
                //     *self.depth,
                //     PathVertexExpr::new(*self.si.inst_id(), *self.si.prim_id(), *self.si.bary()),
                // );
                todo!()
            }
            {
                if self.depth.eq(target_depth) {
                    let (direct, w) = self.handle_surface_light(**ray);
                    self.add_radiance(direct * w);
                }
            }

            if self.depth.ge(self.max_depth) {
                break;
            };
            *self.depth += 1;

            if self.depth.eq(target_depth) {
                let direct_lighting = self.sample_light(u_light);
                if direct_lighting.valid {
                    let shadow_ray = direct_lighting.shadow_ray;
                    if !self.scene.occlude(shadow_ray) {
                        let direct = direct_lighting.irradiance
                            * direct_lighting.bsdf_f
                            * direct_lighting.weight
                            / direct_lighting.pdf;
                        self.add_radiance(direct);
                    }
                }
            }
            let bsdf_sample = self.sample_surface(sampler.next_3d());
            let f = &bsdf_sample.color;
            lc_assert!(f.min().ge(0.0));
            if bsdf_sample.pdf.le(0.0) | !bsdf_sample.valid {
                break_;
            }
            self.mul_beta(f / bsdf_sample.pdf);
            {
                *self.prev_bsdf_pdf = bsdf_sample.pdf;
                *self.prev_ng = self.ng;
                let ro = offset_ray_origin(self.p, face_forward(self.ng, bsdf_sample.wi));
                *ray = Ray::new_expr(
                    ro,
                    bsdf_sample.wi,
                    0.0,
                    1e20,
                    Uint2::expr(self.si.inst_id, self.si.prim_id),
                    Uint2::expr(u32::MAX, u32::MAX),
                );
            }
        }
    }
    // #[tracked]
    // pub fn run_at_depth_replayed(
    //     &self,
    //     ray: Expr<Ray>,
    //     target_depth: Expr<u32>,
    //     sampler: &dyn Sampler,
    //     vertices: VLArrayVar<PathVertex>,
    // ) {
    //     let ray = ray.var();
    //     let u_light = sampler.next_3d();
    //     let prev_p = var!(Float3);
    //     loop_!({
    //         let hit = {
    //             let v = vertices.read(*self.depth);
    //             let hit = v.inst_id().eq(u32::MAX);
    //             if_!(hit, {
    //                 let si = self
    //                     .scene
    //                     .si_from_hitinfo(v.inst_id(), v.prim_id(), v.bary());
    //                 let wo = if_!(self.depth.eq(0), {
    //                     -*ray.d
    //                 }, else {
    //                     (si.geometry().p() - *prev_p).normalize()
    //                 });
    //                 self.set_si(si, wo)
    //             });
    //             hit
    //         };
    //         if_!(!hit, {
    //             let (direct, w) = self.hit_envmap(*ray);
    //             if_!(self.depth.eq(target_depth), {
    //                 self.add_radiance(direct * w);
    //             });
    //             break_();
    //         });
    //         {
    //             if_!(self.depth.eq(target_depth), {
    //                 let (direct, w) = self.handle_surface_light(*ray);
    //                 self.add_radiance(direct * w);
    //             });
    //         }

    //         if_!(self.depth.load().ge(self.max_depth), {
    //             break_();
    //         });
    //         *self.depth.get_mut() += 1;

    //         if_!(self.depth.eq(target_depth), {
    //             let direct_lighting = self.sample_light(u_light);
    //             if_!(direct_lighting.valid, {
    //                 let shadow_ray = direct_lighting.shadow_ray;
    //                 if_!(!self.scene.occlude(shadow_ray), {
    //                     let direct = direct_lighting.irradiance
    //                         * direct_lighting.bsdf_f
    //                         * direct_lighting.weight
    //                         / direct_lighting.pdf;
    //                     self.add_radiance(direct);
    //                 });
    //             });
    //         });
    //         let bsdf_sample = self.sample_surface(sampler.next_3d());
    //         let f = &bsdf_sample.color;
    //         lc_assert!(f.min().ge(0.0));
    //         if_!(bsdf_sample.pdf.le(0.0) | !bsdf_sample.valid, {
    //             break_();
    //         });
    //         self.mul_beta(f / bsdf_sample.pdf);
    //         {
    //             *self.prev_bsdf_pdf.get_mut() = bsdf_sample.pdf;
    //             *self.prev_ng.get_mut() = *self.ng;
    //             *prev_p.get_mut() = *self.p;
    //             let ro = offset_ray_origin(*self.p, face_forward(*self.ng, bsdf_sample.wi));
    //             *ray.get_mut() = RayExpr::new(
    //                 ro,
    //                 bsdf_sample.wi,
    //                 0.0,
    //                 1e20,
    //                 Uint2::expr(*self.si.inst_id(), *self.si.prim_id()),
    //                 Uint2::expr(u32::MAX, u32::MAX),
    //             );
    //         }
    //     })
    // }
}
impl SinglePathMcmc {
    fn sample_dimension(&self, depth: u32) -> u32 {
        4 + (1 + depth) * 3 + 3
    }
    #[tracked]
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
    #[tracked]
    pub fn scalar_contribution(color: &Color) -> Expr<f32> {
        color.max().clamp(0.0f32.expr(), 1e5f32.expr())
        // 1.0f32.expr()
    }
    #[tracked]
    fn evaluate(
        &self,
        scene: &Arc<Scene>,
        filter: PixelFilter,
        eval: &Evaluators,
        independent: &IndependentSampler,
        samples: &Buffer<f32>,
        get_index: impl Fn(Expr<u32>) -> Expr<u32>,
        depth: Expr<u32>,
        is_bootstrap: bool,
    ) -> (Expr<Uint2>, Color, Expr<SampledWavelengths>, Expr<f32>) {
        let mcmc_sampler = McmcSampler::new(
            independent,
            samples,
            get_index,
            self.sample_dimension_device(depth),
        );
        let sampler = if is_bootstrap {
            independent as &dyn Sampler
        } else {
            &mcmc_sampler as &dyn Sampler
        };
        sampler.start();
        let res = scene.camera.resolution().expr();
        let p = sampler.next_2d() * res.cast_f32();
        let p = p.cast_i32().clamp(0, res.cast_i32() - 1);
        let swl = sample_wavelengths(eval.color_repr(), sampler).var();
        let (ray, ray_color, ray_w) =
            scene
                .camera
                .generate_ray(filter, p.cast_u32(), sampler, eval.color_repr(), **swl);
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
            pt.run_at_depth(ray, depth, sampler, None);
            pt.radiance.load() * w
        };
        (p.cast_u32(), l, **swl, Self::scalar_contribution(&l))
    }
    const SAMPLE_BUFFER_AOSOA_SIZE: u32 = 4;

    // layout:
    // [[f32 x 8] x 2] x dims] x (n_chains / 8)
    #[tracked]
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
        self._sample_index_aosoa(chain_id, dim, 0u32.expr())
    }
    fn backup_index_aosoa(&self, chain_id: Expr<u32>, dim: Expr<u32>) -> Expr<u32> {
        self._sample_index_aosoa(chain_id, dim, 1u32.expr())
    }
    fn normalization_factor_correction(&self) -> f32 {
        (self.max_depth - self.min_depth + 1) as f32
    }
    #[tracked]
    fn sample_depth(&self, sampler: &dyn Sampler) -> Expr<u32> {
        let range = self.max_depth + 1 - self.min_depth;
        (self.min_depth + (sampler.next_1d() * range as f32).cast_u32())
            .clamp(self.min_depth.expr(), self.config.max_depth.expr())
    }

    fn bootstrap(&self, scene: &Arc<Scene>, filter: PixelFilter, eval: &Evaluators) -> RenderState {
        let seeds = init_pcg32_buffer(self.device.clone(), self.n_bootstrap);
        let fs = self
            .device
            .create_buffer_from_fn(self.n_bootstrap, |_| 0.0f32);
        let sample_buffer = self
            .device
            .create_buffer(self.sample_dimension(self.config.max_depth) as usize * self.n_chains);
        let bootstrap_kernel = self.device.create_kernel::<fn()>(&track!(|| {
            let i = dispatch_id().x;
            let seed = seeds.var().read(i);
            let sampler = IndependentSampler::from_pcg32(seed.var());

            let depth = self.sample_depth(&sampler);
            // DON'T WRITE TO SAMPLE BUFFER
            // for_range(0u32.expr()..self.sample_dimension_device(depth), |d| {
            //     sample_buffer.write(
            //         self.sample_dimension(self.config.max_depth) * i + d,
            //         sampler.next_1d(),
            //     );
            // });
            let (_p, _l, _swl, f) = self.evaluate(
                scene,
                filter,
                eval,
                &sampler,
                &sample_buffer,
                |d| self.sample_dimension(self.config.max_depth) * i + d,
                depth,
                true,
            );
            fs.var().write(i, f);
        }));
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
            self.sample_dimension(self.max_depth) as usize
                * 2
                * ((self.n_chains + Self::SAMPLE_BUFFER_AOSOA_SIZE as usize - 1)
                    / Self::SAMPLE_BUFFER_AOSOA_SIZE as usize)
                * Self::SAMPLE_BUFFER_AOSOA_SIZE as usize,
        );
        self.device
            .create_kernel::<fn()>(&track!(|| {
                let i = dispatch_id().x;
                let seed_idx = resampled.var().read(i);
                let seed = seeds.var().read(seed_idx);
                let sampler = IndependentSampler::from_pcg32(seed.var());

                let depth = self.sample_depth(&sampler);
                for_range(0u32.expr()..self.sample_dimension_device(depth), |d| {
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
                    false,
                );

                // cpu_dbg!(Float2::expr(f, fs.var().read(seed_idx)));
                let sigma = match &self.method {
                    Method::Kelemen { small_sigma, .. } => *small_sigma,
                    _ => todo!(),
                };
                cur_colors.write(i, l, swl);
                let state = MarkovState::new_expr(i, p, f, 0.0, 0, 0, 0, sigma, depth);
                states.var().write(i, state);
            }))
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
    #[tracked]
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
        let res = scene.camera.resolution().expr().cast_f32();
        // select a mutation strategy
        match self.method {
            Method::Kelemen {
                small_sigma: _,
                large_step_prob,
                adaptive,
                exponential_mutation,
                image_mutation_size,
                image_mutation_prob,
                depth_change_prob,
            } => {
                let chain_id = **state.chain_id;
                let is_large_step = rng.next_1d().lt(large_step_prob);
                let small_step_depth_change = rng.next_1d().lt(depth_change_prob);
                let new_depth = u32::var_zeroed();
                if is_large_step {
                    new_depth.store(self.sample_depth(rng));
                } else {
                    if small_step_depth_change {
                        new_depth.store(self.sample_depth(rng));
                    } else {
                        new_depth.store(state.depth);
                    }
                }

                let mutator = Mutator {
                    is_large_step,
                    method: self.method,
                    small_step_dim_change: small_step_depth_change,
                    old_dim: self.sample_dimension_device(**state.depth),
                    new_dim: self.sample_dimension_device(**new_depth),
                };
                // perform mutation
                mutator.mutate(
                    samples,
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
                    **new_depth,
                    false,
                );
                let proposal_f = f;
                if is_large_step & state.b_cnt().lt(1024 * 1024) {
                    *state.b += proposal_f;
                    *state.b_cnt += 1;
                }
                let cur_f = **state.cur_f;
                let cur_p = **state.cur_pixel;
                let cur_color = cur_color_v.load();
                let accept = select(
                    cur_f.eq(0.0),
                    1.0f32.expr(),
                    (proposal_f / cur_f).clamp(0.0, 1.0),
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
                if rng.next_1d().lt(accept) & state.b_cnt.lt(1024 * 1024) {
                    *state.cur_f = proposal_f;
                    cur_color_v.store(proposal_color);
                    cur_swl.store(proposal_swl);
                    *state.cur_pixel = proposal_p;
                    if !is_large_step {
                        *state.n_accepted += 1;
                    }
                    *state.depth = new_depth;
                } else {
                    mutator.reject(
                        samples,
                        &|dim| self.sample_index_aosoa(chain_id, dim),
                        &|dim| self.backup_index_aosoa(chain_id, dim),
                    );

                }
                if !is_large_step {
                    *state.n_mutations += 1;
                    // if adaptive {
                    //     let r = state.n_accepted().load().cast_f32()
                    //         / state.n_mutations().load().cast_f32();
                    //     const OPTIMAL_ACCEPT_RATE: f32 = 0.234;
                    //     if_!(state.n_mutations().load().gt(50), {
                    //         let new_sigma = state.sigma().load()
                    //             + (r - OPTIMAL_ACCEPT_RATE) / state.n_mutations().load().cast_f32();
                    //         let new_sigma = new_sigma.clamp(1e-5, 0.1);
                    //         if_!(state.chain_id().load().eq(0), {
                    //             cpu_dbg!(Float3::expr(r, state.sigma().load(), new_sigma));
                    //         });
                    //         state.sigma().store(new_sigma);
                    //     });
                    // }
                }
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
        let i = dispatch_id().x;
        let markov_states = render_state.states.var();
        let sampler =
            IndependentSampler::from_pcg32(var!(Pcg32, render_state.rng_states.var().read(i)));
        let state = var!(MarkovState, markov_states.read(i));
        let (cur_color, cur_swl) = render_state.cur_colors.read(i);
        let cur_color_v = ColorVar::new(cur_color);
        let cur_swl_v = def(cur_swl);
        for_range(const_(0)..mutations_per_chain.cast_i32(), |_| {
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
            if let Some(channel) = &session.display {
                film.copy_to_rgba_image(channel.screen_tex(), false);
                channel.notify_update();
            }
        };
        let mut acc_time = 0.0f64;
        let mut stats = RenderStats::default();
        let spp_per_pass = self.config.spp_per_pass;
        let spp = self.config.spp;

        let output_image: luisa::Tex2d<luisa::Float4> = self.device.create_tex2d(
            luisa::PixelStorage::Float4,
            scene.camera.resolution().x,
            scene.camera.resolution().y,
            1,
        );

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
                if session.save_intermediate || session.display.is_some() {
                    reconstruct(film, cnt);
                }
                if session.save_intermediate {
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
        session: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig: {:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        let evaluators = scene.evaluators(color_pipeline, ADMode::None);
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
                &RenderSession {
                    display: session.display.clone(),
                    ..Default::default()
                },
            );
        }
        let render_state = self.bootstrap(&scene, film.filter(), &evaluators);
        self.render_loop(&scene, &evaluators, &render_state, film, session);
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
    let mcmc = SinglePathMcmc::new(device.clone(), config.clone());
    mcmc.render(scene, sampler, color_pipeline, film, options);
}
