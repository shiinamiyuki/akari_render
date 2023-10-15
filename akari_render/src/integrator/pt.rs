use std::{f32::consts::FRAC_1_PI, marker::PhantomData, rc::Rc, sync::Arc, time::Instant};

use luisa::rtx::offset_ray_origin;
use rand::Rng;

use super::{Integrator, RenderSession};
use crate::{
    color::*,
    film::*,
    geometry::*,
    interaction::SurfaceInteraction,
    light::LightEvalContext,
    sampler::*,
    scene::*,
    svm::surface::{diffuse::DiffuseBsdf, *},
    *,
};
use serde::{Deserialize, Serialize};

pub struct PathTracerBase<'a> {
    pub max_depth: Expr<u32>,
    pub use_nee: bool,
    pub force_diffuse: bool,
    pub rr_depth: Expr<u32>,
    pub indirect_only: bool,
    pub radiance: ColorVar,
    pub beta: ColorVar,
    pub color_pipeline: ColorPipeline,
    pub depth: Var<u32>,
    pub prev_bsdf_pdf: Var<f32>,
    pub prev_ng: Var<Float3>,
    pub swl: Var<SampledWavelengths>,
    pub scene: &'a Scene,
}
#[derive(Aggregate)]
pub struct DirectLighting {
    pub irradiance: Color,
    pub wi: Expr<Float3>,
    pub pdf: Expr<f32>,
    pub shadow_ray: Expr<Ray>,
    pub valid: Expr<bool>,
}
impl DirectLighting {
    pub fn invalid(color_pipeline: ColorPipeline) -> Self {
        Self {
            irradiance: Color::zero(color_pipeline.color_repr),
            wi: Expr::<Float3>::zeroed(),
            pdf: 0.0f32.expr(),
            shadow_ray: Expr::<Ray>::zeroed(),
            valid: false.expr(),
        }
    }
}

impl<'a> PathTracerBase<'a> {
    pub fn new(
        scene: &'a Scene,
        color_pipeline: ColorPipeline,
        max_depth: Expr<u32>,
        rr_depth: Expr<u32>,
        use_nee: bool,
        indirect_only: bool,
        swl: Var<SampledWavelengths>,
    ) -> Self {
        Self {
            max_depth,
            use_nee,
            rr_depth,
            indirect_only,
            radiance: ColorVar::zero(color_pipeline.color_repr),
            beta: ColorVar::one(color_pipeline.color_repr),
            scene,
            depth: Var::<u32>::zeroed(),
            prev_bsdf_pdf: Var::<f32>::zeroed(),
            prev_ng: Var::<Float3>::zeroed(),
            swl,
            color_pipeline,
            force_diffuse: false,
        }
    }
    pub fn add_radiance(&self, r: Color) {
        self.radiance
            .store(self.radiance.load() + self.beta.load() * r);
    }
    pub fn mul_beta(&self, r: Color) {
        self.beta.store(self.beta.load() * r);
    }
    pub fn eval_context(&self) -> (LightEvalContext, BsdfEvalContext) {
        let bsdf = BsdfEvalContext {
            color_repr: self.color_pipeline.color_repr,
            ad_mode: ADMode::None,
        };
        let light = LightEvalContext {
            svm: &self.scene.svm,
            meshes: &self.scene.meshes,
            surface_eval_ctx: bsdf,
            color_pipeline: self.color_pipeline,
        };
        (light, bsdf)
    }
    pub fn sample_light(&self, si: SurfaceInteraction, u: Expr<Float3>) -> DirectLighting {
        if self.use_nee {
            track!({
                if !self.indirect_only || self.depth.load().gt(1) {
                    let p = si.p;
                    let ng = si.ng;
                    let pn = PointNormal::new_expr(p, ng);

                    // let sample = eval.light.sample(pn, u, **self.swl);
                    let sample = self.scene.lights.sample_direct(
                        pn,
                        u.x,
                        u.yz(),
                        **self.swl,
                        &self.eval_context().0,
                    );
                    if sample.valid {
                        let wi = sample.wi;
                        // let surface = **self.instance.surface;
                        // let wo = **self.wo;
                        let shadow_ray = sample.shadow_ray.var();
                        *shadow_ray.exclude0 = Uint2::expr(si.inst_id, si.prim_id);
                        // cpu_dbg!(**shadow_ray);
                        DirectLighting {
                            irradiance: sample.li,
                            wi,
                            pdf: sample.pdf,
                            shadow_ray: shadow_ray.load(),
                            valid: true.expr(),
                        }
                    } else {
                        DirectLighting::invalid(self.color_pipeline)
                    }
                } else {
                    DirectLighting::invalid(self.color_pipeline)
                }
            })
        } else {
            DirectLighting::invalid(self.color_pipeline)
        }
    }
    #[tracked]
    pub fn continue_prob(&self) -> (Expr<bool>, Expr<f32>) {
        let depth = **self.depth;
        let beta = self.beta.load();
        if depth.gt(self.rr_depth) {
            let cont_prob = beta.max().clamp(0.0.expr(), 1.0.expr()) * 0.95;
            (true.expr(), cont_prob)
        } else {
            (false.expr(), 1.0f32.expr())
        }
    }
    #[tracked]
    pub fn hit_envmap(&self, ray: Expr<Ray>) -> (Color, Expr<f32>) {
        (Color::zero(self.color_pipeline.color_repr), 0.0f32.expr())
    }
    #[tracked]
    pub fn handle_surface_light(
        &self,
        si: SurfaceInteraction,
        ray: Expr<Ray>,
    ) -> (Color, Expr<f32>) {
        let instance = self.scene.meshes.mesh_instances().read(si.inst_id);
        let depth = **self.depth;

        if instance.light.valid() & (!self.indirect_only | depth.gt(1)) {
            let light_ctx = self.eval_context().0;
            let direct = self.scene.lights.le(ray, si, **self.swl, &light_ctx);
            // cpu_dbg!(direct.flatten());
            if depth.eq(0) | !self.use_nee {
                (direct, 1.0f32.expr())
            } else {
                let pn = {
                    let p = ray.o;
                    let n = self.prev_ng;
                    PointNormal::new_expr(p, **n)
                };
                let light_pdf = self.scene.lights.pdf_direct(si, pn, &light_ctx);
                let w = mis_weight(**self.prev_bsdf_pdf, light_pdf, 1);

                (direct, w)
            }
        } else {
            (Color::zero(self.color_pipeline.color_repr), 0.0f32.expr())
        }
    }
    #[tracked]
    pub fn sample_surface_and_shade_direct(
        &self,
        si: SurfaceInteraction,
        wo: Expr<Float3>,
        direct_lighting: DirectLighting,
        di_occluded: Expr<bool>,
        u_bsdf: Expr<Float3>,
    ) -> BsdfSample {
        let ctx = self.eval_context().1;
        if self.force_diffuse {
            let diffuse = Rc::new(DiffuseBsdf {
                reflectance: Color::one(self.color_pipeline.color_repr)
                    * FRAC_1_PI.expr()
                    * 0.8f32.expr(),
            });
            let closure = SurfaceClosure {
                inner: diffuse,
                frame: si.frame,
            };
            if direct_lighting.valid & !di_occluded {
                let f = closure.evaluate(wo, direct_lighting.wi, **self.swl, &ctx);
                let pdf = closure.pdf(wo, direct_lighting.wi, **self.swl, &ctx);
                let w = mis_weight(direct_lighting.pdf, pdf, 1);
                self.add_radiance(direct_lighting.irradiance * f * w / direct_lighting.pdf);
            };
            let sample = closure.sample(wo, u_bsdf.x, u_bsdf.yz(), self.swl, &ctx);
            sample
        } else {
            let svm = &self.scene.svm;
            svm.dispatch_surface(si.surface, self.color_pipeline, si, **self.swl, |closure| {
                if direct_lighting.valid & !di_occluded {
                    let f = closure.evaluate(wo, direct_lighting.wi, **self.swl, &ctx);
                    let pdf = closure.pdf(wo, direct_lighting.wi, **self.swl, &ctx);
                    let w = mis_weight(direct_lighting.pdf, pdf, 1);
                    self.add_radiance(direct_lighting.irradiance * f * w / direct_lighting.pdf);
                };
                let sample = closure.sample(wo, u_bsdf.x, u_bsdf.yz(), self.swl, &ctx);
                sample
            })
        }
    }
    #[tracked]
    pub fn run_megakernel(&self, ray: Expr<Ray>, sampler: &dyn Sampler) {
        let ray = ray.var();

        loop {
            let si = self.scene.intersect(**ray);
            if !si.valid {
                let (direct, w) = self.hit_envmap(**ray);
                self.add_radiance(direct * w);
                break;
            }
            let wo = -ray.d;
            {
                let (direct, w) = self.handle_surface_light(si, **ray);
                self.add_radiance(direct * w);
            }

            if self.depth.load() >= self.max_depth {
                break;
            }
            *self.depth += 1;

            let direct_lighting = self.sample_light(si, sampler.next_3d());
            let di_occluded = true.var();
            if direct_lighting.valid {
                let shadow_ray = direct_lighting.shadow_ray;
                *di_occluded = self.scene.occlude(shadow_ray);
            }
            let bsdf_sample = self.sample_surface_and_shade_direct(
                si,
                wo,
                direct_lighting,
                **di_occluded,
                sampler.next_3d(),
            );

            let f = &bsdf_sample.color;
            if debug_mode() {
                lc_assert!(f.min().ge(0.0));
            };
            if bsdf_sample.pdf <= 0.0 || !bsdf_sample.valid {
                break;
            }

            self.mul_beta(f / bsdf_sample.pdf);
            let (rr_effective, cont_prob) = self.continue_prob();
            if rr_effective {
                let rr = sampler.next_1d().ge(cont_prob);
                if rr {
                    break;
                }
                self.mul_beta(Color::one(self.color_pipeline.color_repr) / cont_prob);
            }
            {
                *self.prev_bsdf_pdf = bsdf_sample.pdf;
                *self.prev_ng = si.ng;
                let ro = offset_ray_origin(si.p, face_forward(si.ng, bsdf_sample.wi));
                *ray = Ray::new_expr(
                    ro,
                    bsdf_sample.wi,
                    0.0,
                    1e20,
                    Uint2::expr(si.inst_id, si.prim_id),
                    Uint2::expr(u32::MAX, u32::MAX),
                );
            }
        }
    }
}
#[derive(Clone)]
pub struct PathTracer {
    pub device: Device,
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub pixel_offset: Int2,
    pub force_diffuse: bool,
    config: Config,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub force_diffuse: bool,
    pub pixel_offset: [i32; 2],
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
            force_diffuse: false,
            pixel_offset: [0, 0],
        }
    }
}
impl PathTracer {
    pub fn new(device: Device, config: Config) -> Self {
        Self {
            device,
            spp: config.spp,
            max_depth: config.max_depth,
            spp_per_pass: config.spp_per_pass,
            use_nee: config.use_nee,
            rr_depth: config.rr_depth,
            indirect_only: config.indirect_only,
            force_diffuse: config.force_diffuse,
            pixel_offset: Int2::new(config.pixel_offset[0], config.pixel_offset[1]),
            config,
        }
    }
}

pub fn mis_weight(pdf_a: Expr<f32>, pdf_b: Expr<f32>, power: u32) -> Expr<f32> {
    let apply_power = |x: Expr<f32>| {
        let mut p = 1.0f32.expr();
        for _ in 0..power {
            p = track!(p * x);
        }
        p
    };
    let pdf_a = apply_power(pdf_a);
    let pdf_b = apply_power(pdf_b);
    track!(pdf_a / (pdf_a + pdf_b))
}
pub struct VertexType;
impl VertexType {
    pub const INVALID: u32 = 0;
    pub const LAST_HIT_LIGHT: u32 = 1;
    pub const LAST_NEE: u32 = 2;
    pub const INTERIOR: u32 = 3;
}
#[derive(Clone, Copy, Debug, Value)]
#[repr(C)]
pub struct ReconnectionVertex {
    pub bary: Float2,
    pub direct: [f32; 3],
    pub direct_wi: [f32; 3],
    pub indirect: [f32; 3],
    pub wo: [f32; 3],
    pub wi: [f32; 3],
    pub direct_light_pdf: f32,
    pub inst_id: u32,
    pub prim_id: u32,
    pub prev_bsdf_pdf: f32,
    pub bsdf_pdf: f32,
    pub dist: f32,
    pub depth: u32,
    pub type_: u32,
}
impl ReconnectionVertexVar {
    pub fn valid(&self) -> Expr<bool> {
        self.type_.load().ne(VertexType::INVALID)
    }
}
#[derive(Clone, Copy)]
pub struct ReconnectionShiftMapping {
    pub min_dist: Expr<f32>,
    pub is_base_path: Expr<bool>,
    pub vertex: Var<ReconnectionVertex>,
    pub jacobian: Var<f32>,
    pub success: Var<bool>,
    pub min_roughness: Expr<f32>,
}
#[derive(Clone, Copy, Value, Debug)]
#[repr(C)]
pub struct DenoiseFeatures {
    pub albedo: Float3,
    pub normal: Float3,
}
impl PathTracer {
    pub fn radiance(
        &self,
        scene: &Arc<Scene>,
        color_pipeline: ColorPipeline,
        ray: Expr<Ray>,
        swl: Var<SampledWavelengths>,
        sampler: &dyn Sampler,
    ) -> Color {
        let mut pt = PathTracerBase::new(
            scene,
            color_pipeline,
            self.max_depth.expr(),
            self.rr_depth.expr(),
            self.use_nee,
            self.indirect_only,
            swl,
        );
        pt.force_diffuse = self.force_diffuse;
        pt.run_megakernel(ray, sampler);
        pt.radiance.load()
    }
}
impl Integrator for PathTracer {
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
        let sampler_creator = sampler_config.creator(self.device.clone(), &scene, self.spp);
        let kernel =
            self.device.create_kernel::<fn(u32, Int2)>(&track!(
                |spp_per_pass: Expr<u32>, pixel_offset: Expr<Int2>| {
                    set_block_size([16, 16, 1]);
                    let p = dispatch_id().xy();
                    let sampler = sampler_creator.create(p);
                    let sampler = sampler.as_ref();
                    for_range(0u32.expr()..spp_per_pass, |_| {
                        sampler.start();
                        let ip = p.cast_i32();
                        let shifted = ip + pixel_offset;
                        let shifted = shifted
                            .clamp(0, resolution.expr().cast_i32() - 1)
                            .cast_u32();
                        let swl = sample_wavelengths(color_pipeline.color_repr, sampler);
                        let (ray, ray_color, ray_w) = scene.camera.generate_ray(
                            &scene,
                            film.filter(),
                            shifted,
                            sampler,
                            color_pipeline.color_repr,
                            swl,
                        );
                        let swl = swl.var();
                        let l =
                            self.radiance(&scene, color_pipeline, ray, swl, sampler) * ray_color;
                        film.add_sample(p.cast_f32(), &l, **swl, ray_w);
                    });
                }
            ));
        log::info!(
            "Render kernel as {} arguments, {} captures!",
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
            kernel.dispatch(
                [resolution.x, resolution.y, 1],
                &cur_pass,
                &self.pixel_offset,
            );
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
    let pt = PathTracer::new(device.clone(), config.clone());
    pt.render(scene, sampler, color_pipeline, film, options);
}
