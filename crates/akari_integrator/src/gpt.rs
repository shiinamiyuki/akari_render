#![allow(non_snake_case)]

use std::{sync::Arc, time::Instant};

use crate::{
    color::{Color, ColorRepr, FlatColor},
    pt::{self, PathTracer, PathTracerBase, ReconnectionShiftMapping, ReconnectionVertex},
    sampler::*,
    scene::*,
    *,
};
use akari_render::color::{sample_wavelengths, ColorVar, SampledWavelengths};
use serde::{Deserialize, Serialize};

use super::{Integrator, RenderSession};

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
pub enum Reconstruction {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "uniform")]
    Uniform,
    #[serde(rename = "weighted")]
    Weighted,
}
impl Default for Reconstruction {
    fn default() -> Self {
        Self::None
    }
}
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default, crate = "serde")]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub seed: u64,
    pub reconnect: bool,
    pub reconstruction: Reconstruction,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            spp: 256,
            max_depth: 7,
            rr_depth: 5,
            spp_per_pass: 64,
            use_nee: true,
            indirect_only: false,
            seed: 0,
            reconstruction: Default::default(),
            reconnect: true,
        }
    }
}
#[derive(Clone)]
pub struct GradientPathTracer {
    pub device: Device,
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub seed: u64,
    config: Config,
}

impl GradientPathTracer {
    pub fn new(device: Device, config: Config) -> Self {
        Self {
            device,
            spp: config.spp,
            max_depth: config.max_depth,
            spp_per_pass: config.spp_per_pass,
            use_nee: config.use_nee,
            rr_depth: config.rr_depth,
            indirect_only: config.indirect_only,
            seed: config.seed,
            config,
        }
    }
}

struct RenderState<'a> {
    color_pipeline: ColorPipeline,
    primal: &'a Film,
    // Gx: &'a Film,
    // Gy: &'a Film,
    sampler_creator: &'a Box<dyn SamplerCreator>,
    scene: &'a Scene,
}
impl GradientPathTracer {
    #[tracked(crate = "luisa")]
    fn render_one_spp(&self, state: &RenderState) {
        let color_pipeline = state.color_pipeline;
        let sampler_creator = &state.sampler_creator;
        let px = dispatch_id().xy();
        let resolution = state.primal.resolution();
        let scene = state.scene;

        let sampler_backup = sampler_creator.create(px);
        let trace = |is_primary: Expr<bool>,
                     pixel_offset: Expr<Int2>,
                     shift_mapping: Option<ReconnectionShiftMapping>| {
            let l = ColorVar::zero(color_pipeline.color_repr);
            let swl = Var::<SampledWavelengths>::zeroed();
            let ray_w = 0.0f32.var();
            let jacobian = 0.0f32.var();
            let shifted_v = Uint2::var_zeroed();
            outline(|| {
                let sampler = sampler_backup.clone_box();
                let sampler = sampler.as_ref();
                sampler.start();
                if !is_primary {
                    sampler.forget();
                }
                let shifted = px.cast_i32() + pixel_offset;
                let shifted =
                    (shifted + resolution.expr().cast_i32()) % resolution.expr().cast_i32();
                let shifted = shifted.cast_u32();
                *swl = sample_wavelengths(color_pipeline.color_repr, sampler);
                *shifted_v = shifted;
                let (ray, ray_w_) = scene.camera.generate_ray(
                    &scene,
                    state.primal.filter(),
                    shifted,
                    sampler,
                    color_pipeline.color_repr,
                    **swl,
                );
                *ray_w = ray_w_;
                let swl = swl.var();
                let mut pt = PathTracerBase::new(
                    scene,
                    color_pipeline,
                    self.max_depth.expr(),
                    self.rr_depth.expr(),
                    self.use_nee,
                    self.indirect_only,
                    swl,
                );
                let shift_mapping = if let Some(sm) = shift_mapping {
                    pt.need_shift_mapping = true;
                    Some(ReconnectionShiftMapping {
                        is_base_path: is_primary,
                        ..sm
                    })
                } else {
                    None
                };
                pt.run_pt_hybrid_shift_mapping(ray, sampler, shift_mapping);
                l.store(pt.radiance.load());
                if let Some(sm) = shift_mapping {
                    *jacobian = sm.jacobian;
                } else {
                    *jacobian = 1.0f32.expr();
                }
            });
            (**shifted_v, l.load(), **swl, **ray_w, **jacobian)
        };
        let shift_mapping = if self.config.reconnect {
            Some(ReconnectionShiftMapping {
                min_dist: 0.03f32.expr(),
                min_roughness: 0.0f32.expr(),
                is_base_path: true.expr(),
                vertex: ReconnectionVertex::var_zeroed(),
                jacobian: 0.0f32.var(),
                success: false.var(),
            })
        } else {
            None
        };
        let (_, primary_l, primary_swl, primay_ray_w, _) =
            trace(true.expr(), Int2::expr(0, 0), shift_mapping);
        let get_offset = |i: Expr<u32>| {
            if i == 0 {
                Int2::expr(1, 0)
            } else if i == 1 {
                Int2::expr(0, 1)
            } else if i == 2 {
                Int2::expr(-1, 0)
            } else {
                Int2::expr(0, -1)
            }
        };
        // state.primal.add_sample(
        //     px.cast_f32(),
        //     &primary_l,
        //     primary_swl,
        //     primay_ray_w, // * mis_w_primary,
        // );
        for i in 0..4u32 {
            let offset = get_offset(i);
            let shift_mapping = shift_mapping.map(|sm| ReconnectionShiftMapping {
                is_base_path: false.expr(),
                success: false.var(),
                jacobian: 0.0f32.var(),
                ..sm
            });
            let (shifted, l, swl, ray_w, jacobian) = trace(false.expr(), offset, shift_mapping);

            let mis_w_primary = 1.0 / (1.0 + jacobian);
            let mis_w_shifted = jacobian / (1.0 + jacobian);

            state.primal.add_sample(
                px.cast_f32(),
                &primary_l,
                primary_swl,
                primay_ray_w * mis_w_primary,
            );
            state
                .primal
                .add_sample(shifted.cast_f32(), &(l * jacobian), swl, ray_w * mis_w_shifted);
        }
        sampler_backup.forget();
    }
}
impl Integrator for GradientPathTracer {
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
            "Resolution {}x{}\nconfig:{:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        // let primal = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        // let Gx = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        // let Gy = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        let sampler_creator = sampler_config.creator(self.device.clone(), &scene, self.spp);
        let kernel = self
            .device
            .create_kernel::<fn(u32)>(&|spp_per_pass: Expr<u32>| {
                let state = RenderState {
                    primal: film,
                    // Gx: &Gx,
                    // Gy: &Gy,
                    sampler_creator: &sampler_creator,
                    color_pipeline,
                    scene: &scene,
                };
                for_range(0u32.expr()..spp_per_pass, |_| {
                    self.render_one_spp(&state);
                });
            });

        log::info!(
            "Render kernel has {} arguments, {} captures!",
            kernel.num_arguments(),
            kernel.num_capture_arguments()
        );
        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        let mut acc_time = 0.0;
        let update = || {
            if let Some(channel) = &session.display {
                film.copy_to_rgba_image(channel.screen_tex(), false);
                channel.notify_update();
            }
        };

        while cnt < self.spp {
            let cur_pass = (self.spp - cnt).min(self.spp_per_pass);
            let tic = Instant::now();
            kernel.dispatch([resolution.x, resolution.y, 1], &cur_pass);
            let toc = Instant::now();
            acc_time += toc.duration_since(tic).as_secs_f64();
            update();
            progress.inc(cur_pass as u64);
            cnt += cur_pass;
        }

        progress.finish();
        log::info!("Rendering finished in {:.2}s", acc_time);
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
    let gpt = GradientPathTracer::new(device.clone(), config.clone());
    gpt.render(scene, sampler, color_pipeline, film, options);
}
