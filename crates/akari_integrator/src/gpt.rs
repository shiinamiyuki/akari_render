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

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
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
    pub stride: u32,
    pub separate_weights: bool,
    pub reconstruction_iter: u32,
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
            separate_weights: false,
            reconstruction_iter: 30,
            stride: 1,
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
#[allow(non_snake_case)]
struct RenderState<'a> {
    color_pipeline: ColorPipeline,
    primal: &'a Film,
    Gx: &'a Film,
    Gy: &'a Film,
    sampler_creator: &'a Box<dyn SamplerCreator>,
    scene: &'a Scene,
}
impl GradientPathTracer {
    /// returns (is_grad_x, sign, offset)
    #[tracked(crate = "luisa")]
    fn get_grad_film_offset(&self, i: Expr<u32>) -> (Expr<bool>, Expr<f32>, Expr<Uint2>) {
        if i == 0 {
            (true.expr(), 1.0f32.expr(), Uint2::expr(1, 0))
        } else if i == 1 {
            (false.expr(), 1.0f32.expr(), Uint2::expr(0, 1))
        } else if i == 2 {
            (true.expr(), -1.0f32.expr(), Uint2::expr(0, 0))
        } else {
            (false.expr(), -1.0f32.expr(), Uint2::expr(0, 0))
        }
    }

    #[tracked(crate = "luisa")]
    fn get_shifted(&self, resolution: Expr<Uint2>, px: Expr<Uint2>, i: Expr<u32>) -> Expr<Uint2> {
        let off = if i == 0 {
            Int2::expr(1, 0)
        } else if i == 1 {
            Int2::expr(0, 1)
        } else if i == 2 {
            Int2::expr(-1, 0)
        } else {
            Int2::expr(0, -1)
        };
        let reflect = |x: Expr<i32>, r: Expr<u32>| {
            if x < 0 {
                (-x).as_u32()
            } else if x.as_u32() >= r {
                r - (x.as_u32() - r) - 1
            } else {
                x.as_u32()
            }
        };
        let q = px.cast_i32() + off * self.config.stride as i32;
        Uint2::expr(reflect(q.x, resolution.x), reflect(q.y, resolution.y))
    }
    #[tracked(crate = "luisa")]
    fn render_one_spp(&self, state: &RenderState) {
        let color_pipeline = state.color_pipeline;
        let sampler_creator = &state.sampler_creator;
        let px = dispatch_id().xy();
        let resolution = state.primal.resolution();
        let scene = state.scene;

        let sampler_backup = sampler_creator.create(px);
        let trace = |is_primary: Expr<bool>,
                     pixel: Expr<Uint2>,
                     shift_mapping: Option<&ReconnectionShiftMapping>| {
            let l: ColorVar = ColorVar::zero(color_pipeline.color_repr);
            let swl = Var::<SampledWavelengths>::zeroed();
            let ray_w = 0.0f32.var();
            let jacobian = 0.0f32.var();
            let reconnect = ColorVar::zero(color_pipeline.color_repr);
            outline(|| {
                let sampler = sampler_backup.clone_box();
                let sampler = sampler.as_ref();
                sampler.start();

                sampler.forget();

                *swl = sample_wavelengths(color_pipeline.color_repr, sampler);
                let (ray, ray_w_) = scene.camera.generate_ray(
                    &scene,
                    state.primal.filter(),
                    pixel,
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
                if let Some(sm) = &shift_mapping {
                    pt.need_shift_mapping = true;
                    *sm.is_base_path = is_primary;
                };
                pt.run_pt_hybrid_shift_mapping(ray, sampler, shift_mapping, None);

                if let Some(sm) = shift_mapping {
                    if self.config.separate_weights {
                        l.store(pt.base_replay_throughput.load());
                    } else {
                        l.store(pt.radiance.load());
                    }
                    *jacobian = sm.jacobian;
                    reconnect.store(pt.radiance.load() - l.load());
                } else {
                    l.store(pt.radiance.load());
                    *jacobian = 1.0f32.expr();
                }
            });
            (l.load(), reconnect.load(), **swl, **ray_w, **jacobian)
        };
        let vertex = ReconnectionVertex::var_zeroed();
        let shift_mapping = if self.config.reconnect {
            Some(ReconnectionShiftMapping {
                min_dist: 0.03f32.expr(),
                min_roughness: 0.2f32.expr(),
                is_base_path: true.var(),
                read_vertex: Box::new(|| **vertex),
                write_vertex: Box::new(|v| vertex.store(v)),
                jacobian: 0.0f32.var(),
                success: false.var(),
            })
        } else {
            None
        };
        let (primary_l, primary_reconnect, primary_swl, primay_ray_w, _) =
            trace(true.expr(), px, shift_mapping.as_ref());
        if self.config.reconstruction != Reconstruction::None {
            state.primal.add_splat(
                px.cast_f32(),
                &(primary_l + primary_reconnect),
                primary_swl,
                primay_ray_w,
            );
        }
        {
            // let shift_mapping = shift_mapping.map(|sm| ReconnectionShiftMapping {
            //     is_base_path: false.expr(),
            //     success: false.var(),
            //     jacobian: 0.0f32.var(),
            //     ..sm
            // });
            // let (_shifted, l, shifted_reconnect, _, _, _) =
            //     trace(false.expr(), Int2::expr(0, 0), shift_mapping);
            // if let Some(sm) = shift_mapping {
            //     state.primal.add_sample(
            //         px.cast_f32(),
            //         &(Color::one(color_pipeline.color_repr) * **sm.jacobian),
            //         primary_swl,
            //         primay_ray_w, // * mis_w_primary,
            //     );
            // }
            // state.primal.add_sample(
            //     px.cast_f32(),
            //     &((primary_l - l) * 1.0f32.expr()),
            //     primary_swl,
            //     primay_ray_w, // * mis_w_primary,
            // );
            // let primary_l = primary_l.as_rgb();
            // let l = l.as_rgb();
            // let primary_reconnect = primary_reconnect.as_rgb();
            // let shifted_reconnect = shifted_reconnect.as_rgb();
            // state.primal.add_sample(
            //     px.cast_f32(),
            //     &((primary_reconnect - shifted_reconnect) * 1.0f32.expr()),
            //     primary_swl,
            //     primay_ray_w, // * mis_w_primary,
            // );
            // if (l - primary_l).abs().reduce_max() > 0.01 {
            //     device_log!("P mismatch: {} vs {} at {}", primary_l, l, px.cast_f32());
            // }
            // if (shifted_reconnect - primary_reconnect).abs().reduce_max() > 0.01 {
            //     device_log!(
            //         "R mismatch: {} vs {} at {}",
            //         primary_reconnect,
            //         shifted_reconnect,
            //         px.cast_f32()
            //     );
            // }
        }
        for i in 0..4u32 {
            let shifted = self.get_shifted(resolution.expr(), px, i);
            if let Some(sm) = &shift_mapping {
                *sm.is_base_path = false;
                *sm.success = false;
                *sm.jacobian = 0.0f32;
            };
            let (l, shifted_reconnect, swl, ray_w, jacobian) =
                trace(false.expr(), shifted, shift_mapping.as_ref());
            if self.config.reconstruction == Reconstruction::None {
                let (mis_w_primary, mis_w_shifted) = if shift_mapping.as_ref().unwrap().success {
                    (1.0 / (1.0 + jacobian), 1.0 / (1.0 + jacobian))
                } else {
                    (1.0f32.expr(), 0.0f32.expr())
                };
                if self.config.separate_weights {
                    state.primal.add_splat(
                        px.cast_f32(),
                        &(primary_l * 0.5f32.expr() + primary_reconnect * mis_w_primary),
                        primary_swl,
                        primay_ray_w,
                    );
                    state.primal.add_splat(
                        shifted.cast_f32(),
                        &(l * 0.5f32.expr() + shifted_reconnect * mis_w_shifted * jacobian),
                        swl,
                        ray_w,
                    );
                } else {
                    state.primal.add_splat(
                        px.cast_f32(),
                        &(primary_l * mis_w_primary),
                        primary_swl,
                        primay_ray_w,
                    );
                    state.primal.add_splat(
                        shifted.cast_f32(),
                        &(l * mis_w_shifted * jacobian),
                        swl,
                        ray_w,
                    );
                }
            } else {
                let grad = if let Some(shift_mapping) = &shift_mapping {
                    if self.config.separate_weights {
                        let grad_reconnect = if shift_mapping.success {
                            (shifted_reconnect * jacobian - primary_reconnect) / (1.0 + jacobian)
                        } else {
                            Color::zero(color_pipeline.color_repr) - primary_reconnect
                        };
                        let grad = (l - primary_l) * 0.5f32.expr();
                        grad + grad_reconnect
                    } else {
                        if shift_mapping.success {
                            (l * jacobian - primary_l) / (1.0 + jacobian)
                        } else {
                            Color::zero(color_pipeline.color_repr) - primary_l
                        }
                    }
                } else {
                    (l - primary_l) * 0.5f32.expr()
                };

                let (is_grad_x, sign, offset) = self.get_grad_film_offset(i);
                if is_grad_x {
                    state
                        .Gx
                        .add_splat(px.cast_f32() + offset.cast_f32(), &grad, swl, ray_w * sign);
                } else {
                    state
                        .Gy
                        .add_splat(px.cast_f32() + offset.cast_f32(), &grad, swl, ray_w * sign);
                }
            }
        }
        sampler_backup.start();
    }
}
struct Films {
    primal: Film,
    Gx: Film,
    Gy: Film,
}
impl Films {
    fn new(device: Device, resolution: Uint2, repr: FilmColorRepr, filter: PixelFilter) -> Self {
        let films = Self {
            primal: Film::new(device.clone(), resolution, repr, filter),
            Gx: Film::new(
                device.clone(),
                Uint2::new(resolution.x + 1, resolution.y + 1),
                repr,
                filter,
            ),
            Gy: Film::new(
                device.clone(),
                Uint2::new(resolution.x + 1, resolution.y + 1),
                repr,
                filter,
            ),
        };
        films.primal.clear();
        films.Gx.clear();
        films.Gy.clear();
        films
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
        let tmp = Films::new(self.device.clone(), resolution, film.repr(), film.filter());
        let mut acc = Films::new(self.device.clone(), resolution, film.repr(), film.filter());
        let sqr = Films::new(self.device.clone(), resolution, film.repr(), film.filter());
        let sampler_creator = sampler_config.creator(self.device.clone(), &scene, self.spp);
        let kernel = self.device.create_kernel::<fn()>(&|| {
            let state = RenderState {
                primal: &tmp.primal,
                Gx: &tmp.Gx,
                Gy: &tmp.Gy,
                sampler_creator: &sampler_creator,
                color_pipeline,
                scene: &scene,
            };

            self.render_one_spp(&state);
        });

        log::info!(
            "Render kernel has {} arguments, {} captures!",
            kernel.num_arguments(),
            kernel.num_capture_arguments()
        );
        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        let mut acc_time = 0.0;
        let update_kernel = self.device.create_kernel::<fn()>(&track!(|| {
            let px = dispatch_id().xy();
            let resolution = tmp.primal.resolution().expr();
            let px_idx = px.x + px.y * resolution.x;
            let grad_px_idx = px.x + px.y * (resolution.x + 1);

            for c in 0u32..film.repr().nvalues() as u32 {
                let primal_splat_offset =
                    tmp.primal.splat_offset() + px_idx * film.repr().nvalues() as u32 + c;
                let grad_splat_offset =
                    tmp.Gx.splat_offset() + grad_px_idx * film.repr().nvalues() as u32 + c;
                let v = tmp.primal.data().read(primal_splat_offset);

                if self.config.reconstruction == Reconstruction::None {
                    film.data().atomic_fetch_add(primal_splat_offset, v * 0.25);
                } else {
                    let gx = tmp.Gx.data().read(grad_splat_offset);
                    let gy = tmp.Gy.data().read(grad_splat_offset);

                    acc.primal.data().atomic_fetch_add(primal_splat_offset, v);
                    acc.Gx.data().atomic_fetch_add(grad_splat_offset, gx);
                    acc.Gy.data().atomic_fetch_add(grad_splat_offset, gy);

                    sqr.primal
                        .data()
                        .atomic_fetch_add(primal_splat_offset, v * v);
                    sqr.Gx.data().atomic_fetch_add(grad_splat_offset, gx * gx);
                    sqr.Gy.data().atomic_fetch_add(grad_splat_offset, gy * gy);
                }

                tmp.primal.data().write(primal_splat_offset, 0.0f32.expr());
                tmp.Gx.data().write(grad_splat_offset, 0.0f32.expr());
                tmp.Gy.data().write(grad_splat_offset, 0.0f32.expr());
            }
        }));
        let update = |film: &mut Film| {
            if self.config.reconstruction == Reconstruction::None {
                film.set_splat_scale(1.0 / self.spp as f32);
            }
            if let Some(channel) = &session.display {
                film.copy_to_rgba_image(channel.screen_tex(), false);
                channel.notify_update();
            }
        };

        while cnt < self.spp {
            let cur_pass = (self.spp - cnt).min(self.spp_per_pass);
            let tic = Instant::now();
            self.device.default_stream().with_scope(|s| {
                let mut cmds = Vec::new();
                for _ in 0..cur_pass {
                    cmds.push(kernel.dispatch_async([resolution.x, resolution.y, 1]));
                    cmds.push(update_kernel.dispatch_async([resolution.x, resolution.y, 1]));
                }
                s.submit(cmds);
            });
            cnt += cur_pass;
            let toc = Instant::now();
            acc_time += toc.duration_since(tic).as_secs_f64();
            update(film);
            progress.inc(cur_pass as u64);
        }
        progress.finish();
        let recon_old = Film::new(self.device.clone(), resolution, film.repr(), film.filter());
        if self.config.reconstruction != Reconstruction::None {
            log::info!("Reconstructing...");
            self.device
                .create_kernel::<fn(u32)>(&track!(|spp: Expr<u32>| {
                    let px = dispatch_id().xy();
                    let px_idx = px.x + px.y * resolution.x;
                    for c in 0u32..film.repr().nvalues() as u32 {
                        let primal_splat_offset =
                            recon_old.splat_offset() + px_idx * film.repr().nvalues() as u32 + c;
                        let v = acc.primal.data().read(primal_splat_offset);
                        recon_old
                            .data()
                            .write(primal_splat_offset, v / spp.as_f32());
                    }
                }))
                .dispatch([resolution.x, resolution.y, 1], &self.config.spp);
            let eps = 0.01;
            let primal_var_scaling = (0..self.config.reconstruction_iter)
                .map(|i| 1.0 / (eps + 1.0 + 4.0 * 0.5f32.powi(i as i32)))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            // compute prefix multiplication
            let mut prefix_mul = vec![1.0f32; self.config.reconstruction_iter as usize];
            for i in 1..self.config.reconstruction_iter as usize {
                prefix_mul[i] = prefix_mul[i - 1] * primal_var_scaling[i - 1];
            }
            let primal_var_scaling = self.device.create_buffer_from_slice(&prefix_mul);
            let recons_kernel =
                self.device
                    .create_kernel::<fn(u32, u32)>(&|it: Expr<u32>, spp: Expr<u32>| {
                        track!({
                            let spp = spp.as_f32();
                            let px = dispatch_id().xy();
                            let get_px_idx = |px: Expr<Uint2>| {
                                let px_idx = px.x + px.y * resolution.x;
                                px_idx
                            };
                            let get_grad_px_idx = |px: Expr<Uint2>| {
                                let px_idx = px.x + px.y * (resolution.x + 1);
                                px_idx
                            };

                            for c in 0u32..film.repr().nvalues() as u32 {
                                let primal_splat_offset = recon_old.splat_offset()
                                    + get_px_idx(px) * film.repr().nvalues() as u32
                                    + c;
                                let primal = recon_old.data().read(primal_splat_offset);
                                let primal2 = sqr.primal.data().read(primal_splat_offset) / spp;
                                let primal_var = (primal2
                                    - (acc.primal.data().read(primal_splat_offset) / spp).sqr())
                                .max_(1e-6f32.expr())
                                    / spp;
                                let v = 0.0f32.var();
                                let primal_weight =
                                    if self.config.reconstruction == Reconstruction::Uniform {
                                        1.0f32.expr()
                                    } else {
                                        (primal_var * primal_var_scaling.read(it)).recip()
                                    };
                                let sum_w = 0.0f32.var();
                                *v += primal * primal_weight;
                                *sum_w += primal_weight;
                                for i in 0..4u32 {
                                    let (is_grad_x, sign, offset) = self.get_grad_film_offset(i);
                                    let shifted = self.get_shifted(resolution.expr(), px, i);
                                    let grad = if is_grad_x {
                                        acc.Gx.data().read(
                                            acc.Gx.splat_offset()
                                                + get_grad_px_idx(px + offset)
                                                    * film.repr().nvalues() as u32
                                                + c,
                                        ) / spp
                                    } else {
                                        acc.Gy.data().read(
                                            acc.Gy.splat_offset()
                                                + get_grad_px_idx(px + offset)
                                                    * film.repr().nvalues() as u32
                                                + c,
                                        ) / spp
                                    };
                                    let grad2 = if is_grad_x {
                                        sqr.Gx.data().read(
                                            sqr.Gx.splat_offset()
                                                + get_grad_px_idx(px + offset)
                                                    * film.repr().nvalues() as u32
                                                + c,
                                        ) / spp
                                    } else {
                                        sqr.Gy.data().read(
                                            sqr.Gy.splat_offset()
                                                + get_grad_px_idx(px + offset)
                                                    * film.repr().nvalues() as u32
                                                + c,
                                        ) / spp
                                    };
                                    let grad_var =
                                        ((grad2 - grad.sqr()) / spp).max_(1e-6f32.expr());
                                    let primal = recon_old.data().read(
                                        recon_old.splat_offset()
                                            + get_px_idx(shifted) * film.repr().nvalues() as u32
                                            + c,
                                    );
                                    let var = primal_var + grad_var;
                                    let w = if self.config.reconstruction == Reconstruction::Uniform
                                    {
                                        1.0f32.expr()
                                    } else {
                                        var.recip()
                                    };
                                    *v += (primal - sign * grad) * w;
                                    *sum_w += w;
                                }
                                film.data().write(primal_splat_offset, v / sum_w);
                            }
                        })
                    });
            let progress =
                util::create_progess_bar(self.config.reconstruction_iter as usize, "iters");
            for i in 0..self.config.reconstruction_iter {
                recons_kernel.dispatch([resolution.x, resolution.y, 1], &i, &self.config.spp);
                film.data().copy_to_buffer(&recon_old.data());
                progress.inc(1);
            }
            progress.finish();
        }

        // save intermediate films
        if self.config.reconstruction != Reconstruction::None {
            let image: Tex2d<Float4> = self.device.create_tex2d(
                PixelStorage::Float4,
                scene.camera.resolution().x,
                scene.camera.resolution().y,
                1,
            );
            let grad_image: Tex2d<Float4> = self.device.create_tex2d(
                PixelStorage::Float4,
                scene.camera.resolution().x + 1,
                scene.camera.resolution().y + 1,
                1,
            );
            acc.primal.set_splat_scale(1.0 / self.spp as f32);
            acc.Gx.set_splat_scale(1.0 / self.spp as f32);
            acc.Gy.set_splat_scale(1.0 / self.spp as f32);
            acc.primal.copy_to_rgba_image(&image, true);
            util::write_image(&image, &format!("output/gpt_primal.exr"));
            acc.Gx.copy_to_rgba_image(&grad_image, true);
            util::write_image(&grad_image, &format!("output/gpt_gx.exr"));
            acc.Gy.copy_to_rgba_image(&grad_image, true);
            util::write_image(&grad_image, &format!("output/gpt_gy.exr"));
        }

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
