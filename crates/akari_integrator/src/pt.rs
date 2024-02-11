//! Basic megakernel path tracer

use std::{f32::consts::FRAC_1_PI, fs::File, io::BufWriter, rc::Rc, sync::Arc, time::Instant};

use akari_render::svm::ShaderRef;
use luisa::rtx::offset_ray_origin;

use super::{Integrator, RenderSession};
use crate::{
    color::*,
    geometry::*,
    interaction::SurfaceInteraction,
    light::LightEvalContext,
    sampler::*,
    svm::surface::{diffuse::DiffuseBsdf, *},
    *,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Aggregate)]
#[luisa(crate = "luisa")]
pub struct DenoisingFeatures {
    pub first_hit_albedo: ColorVar,
    pub first_hit_roughness: Var<f32>,
    pub first_hit_normal: Var<Float3>,
    pub first_hit: Var<SurfaceHit>,
}

pub struct PathTracerBase<'a> {
    pub max_depth: Expr<u32>,
    pub use_nee: bool,
    pub force_diffuse: bool,
    pub rr_depth: Expr<u32>,
    pub indirect_only: bool,
    pub radiance: ColorVar,
    pub beta: ColorVar,
    pub reconnect_radiance: ColorVar,
    pub reconnect_beta: ColorVar,
    pub base_replay_throughput: ColorVar,
    pub color_pipeline: ColorPipeline,
    pub depth: Var<u32>,
    pub prev_bsdf_pdf: Var<f32>,
    pub prev_ng: Var<Float3>,
    pub prev_p: Var<Float3>,
    pub prev_roughness: Var<f32>,
    pub swl: Var<SampledWavelengths>,
    pub scene: &'a Scene,
    pub need_shift_mapping: bool,
    pub debug_depth: Option<Expr<u32>>,
    pub min_reconnect_depth: u32,
    /// disable nee on first hit during shift mapping
    pub shift_mapping_no_first_nee: bool,
    /// disable all nee during shift mapping, does not include nee in the base path
    pub shift_mapping_no_nee: bool,
    pub denoising: Option<DenoisingFeatures>,
    pub enable_debug: Var<bool>,
    pub clamp_indirect: f32,
}
#[derive(Aggregate, Copy, Clone)]
#[luisa(crate = "luisa")]
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

#[derive(Copy, Clone, Value)]
#[repr(C, align(16))]
#[luisa(crate = "luisa")]
pub struct SurfaceHit {
    pub inst_id: u32,
    pub prim_id: u32,
    pub bary: Float2,
}

#[derive(Aggregate, Copy, Clone)]
#[luisa(crate = "luisa")]
pub struct BsdfDenoisingFeatures {
    pub albedo: Color,
    pub roughness: Expr<f32>,
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
            reconnect_radiance: ColorVar::zero(color_pipeline.color_repr),
            reconnect_beta: ColorVar::one(color_pipeline.color_repr),
            base_replay_throughput: ColorVar::zero(color_pipeline.color_repr),
            scene,
            depth: Var::<u32>::zeroed(),
            prev_bsdf_pdf: Var::<f32>::zeroed(),
            prev_ng: Var::<Float3>::zeroed(),
            prev_p: Float3::var_zeroed(),
            prev_roughness: Var::<f32>::zeroed(),
            swl,
            color_pipeline,
            force_diffuse: false,
            need_shift_mapping: false,
            debug_depth: None,
            denoising: None,
            min_reconnect_depth: 1,
            shift_mapping_no_first_nee: false,
            shift_mapping_no_nee: false,
            enable_debug: false.var(),
            clamp_indirect: 1000.0,
        }
    }
    #[tracked(crate = "luisa")]
    pub fn add_radiance(&self, r: Color) {
        let f = || {
            self.radiance
                .store(self.radiance.load() + self.beta.load() * r);
        };
        if let Some(debug_depth) = self.debug_depth {
            if self.depth == debug_depth {
                f();
            }
        } else {
            f();
        }
        if self.need_shift_mapping {
            self.reconnect_radiance
                .store(self.reconnect_radiance.load() + self.reconnect_beta.load() * r);
        }
    }
    pub fn mul_beta(&self, r: Color) {
        self.beta.store(self.beta.load() * r);
        if self.need_shift_mapping {
            self.reconnect_beta.store(self.reconnect_beta.load() * r);
        }
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
    #[tracked(crate = "luisa")]
    pub fn compute_contibue_prob(&self, depth: Expr<u32>, beta: Color) -> (Expr<bool>, Expr<f32>) {
        if depth.gt(self.rr_depth) {
            let cont_prob = beta.reduce_max().clamp(0.0.expr(), 1.0.expr()) * 0.95;
            (true.expr(), cont_prob)
        } else {
            (false.expr(), 1.0f32.expr())
        }
    }
    #[tracked(crate = "luisa")]
    pub fn continue_prob(&self) -> (Expr<bool>, Expr<f32>) {
        let depth = **self.depth;
        let beta = self.beta.load();
        self.compute_contibue_prob(depth, beta)
    }
    #[tracked(crate = "luisa")]
    pub fn hit_envmap(&self, _ray: Expr<Ray>) -> (Color, Expr<f32>) {
        (Color::zero(self.color_pipeline.color_repr), 0.0f32.expr())
    }
    #[tracked(crate = "luisa")]
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
    #[tracked(crate = "luisa")]
    fn dispatch_surface<R: Aggregate>(
        &self,
        shader_kind: Option<u32>,
        shader_ref: Expr<ShaderRef>,
        si: SurfaceInteraction,
        swl: Expr<SampledWavelengths>,
        f: &impl Fn(&SurfaceClosure) -> R,
    ) -> R {
        if self.force_diffuse {
            let diffuse = Rc::new(DiffuseBsdf {
                reflectance: Color::one(self.color_pipeline.color_repr)
                    * FRAC_1_PI.expr()
                    * 0.8f32.expr(),
            });
            let closure = SurfaceClosure {
                inner: diffuse,
                frame: si.frame,
                ng: si.ng,
            };
            f(&closure)
        } else {
            let svm = &self.scene.svm;
            if let Some(shader_kind) = shader_kind {
                svm.dispatch_surface_single_kind(
                    shader_kind,
                    shader_ref,
                    self.color_pipeline,
                    si,
                    swl,
                    f,
                )
            } else {
                svm.dispatch_surface(shader_ref, self.color_pipeline, si, swl, f)
            }
        }
    }
    #[tracked(crate = "luisa")]
    pub fn sample_surface_and_shade_direct(
        &self,
        shader_kind: Option<u32>,
        si: SurfaceInteraction,
        wo: Expr<Float3>,
        direct_lighting: DirectLighting,
        u_bsdf: Expr<Float3>,
    ) -> (BsdfSample, BsdfDenoisingFeatures, Color) {
        let ctx = self.eval_context().1;

        let sample_and_shade = |closure: &SurfaceClosure| {
            let direct = if direct_lighting.valid {
                let (f, pdf) = closure.evaluate(wo, direct_lighting.wi, **self.swl, &ctx);
                let w = mis_weight(direct_lighting.pdf, pdf, 1);
                direct_lighting.irradiance * f * w / direct_lighting.pdf
            } else {
                Color::zero(self.color_pipeline.color_repr)
            };

            let sample = closure.sample(wo, u_bsdf.x, u_bsdf.yz(), self.swl, &ctx);
            let albedo =
                closure.albedo(wo, **self.swl, &ctx) + closure.emission(wo, **self.swl, &ctx);
            let roughness = closure.roughness(wo, u_bsdf.x, **self.swl, &ctx);
            (sample, BsdfDenoisingFeatures { albedo, roughness }, direct)
        };
        self.dispatch_surface(shader_kind, si.surface, si, **self.swl, &sample_and_shade)
    }
    #[tracked(crate = "luisa")]
    pub fn run_megakernel(&self, ray: Expr<Ray>, sampler: &dyn Sampler) {
        self.run_pt_hybrid_shift_mapping(ray, sampler, None, None)
    }
    #[tracked(crate = "luisa")]
    pub fn run_pt_hybrid_shift_mapping(
        &self,
        ray: Expr<Ray>,
        sampler: &dyn Sampler,
        shift_mapping: Option<&ReconnectionShiftMapping>,
        load_cached_first_hit: Option<&dyn Fn() -> Expr<SurfaceHit>>,
    ) {
        if let Some(sm) = shift_mapping {
            assert!(self.need_shift_mapping);
            if !sm.is_base_path {
                *sm.success = false;
                *sm.jacobian = 0.0f32.expr();
            }
        }
        let rejected = false.var();
        let ray = ray.var();
        let always_reconnect_first_hit = false.var();
        if let Some(sm) = shift_mapping {
            if !sm.is_base_path {
                if (sm.min_dist <= 0.0) & (sm.min_roughness <= 0.0) {
                    *always_reconnect_first_hit = true;
                }
            }
        }
        loop {
            let disable_nee = false.var();
            let disable_first_nee = false.var();
            if let Some(sm) = shift_mapping {
                *disable_nee = !sm.is_base_path
                    & (self.shift_mapping_no_nee
                        | (self.shift_mapping_no_first_nee & (self.depth == 0)));
                *disable_first_nee =
                    !sm.is_base_path & (self.shift_mapping_no_first_nee & (self.depth == 0));
            }
            let si = if load_cached_first_hit.is_some() {
                if self.depth == 0 {
                    let cached_first_hit = (load_cached_first_hit.unwrap())();
                    if cached_first_hit.inst_id == u32::MAX {
                        SurfaceInteraction::invalid()
                    } else {
                        self.scene.surface_interaction(
                            cached_first_hit.inst_id,
                            cached_first_hit.prim_id,
                            cached_first_hit.bary,
                        )
                    }
                } else {
                    self.scene.intersect(**ray)
                }
            } else {
                self.scene.intersect(**ray)
            };
            if !si.valid {
                // if self.enable_debug {
                //     device_log!("!si.valid; d:{}", **self.depth);
                // }
                if !disable_nee {
                    let (direct, w) = self.hit_envmap(**ray);
                    self.add_radiance(direct * w);
                    if self.depth == 0 {
                        if let Some(df) = self.denoising {
                            *df.first_hit.inst_id = u32::MAX;
                            *df.first_hit.prim_id = u32::MAX;
                        }
                    }
                }
                break;
            }
            if self.depth == 0 {
                if let Some(df) = self.denoising {
                    *df.first_hit_normal = si.ng;
                    *df.first_hit = SurfaceHit::from_comps_expr(SurfaceHitComps {
                        inst_id: si.inst_id,
                        prim_id: si.prim_id,
                        bary: si.bary,
                    });
                }
            }

            let wo = -ray.d;

            if !disable_first_nee {
                let (direct, w) = self.handle_surface_light(si, **ray);
                // let w = (!disable_nee | (self.depth == 0)).select(w, 1.0f32.expr(), w);
                self.add_radiance(direct * w);
            }
            if self.depth == 0 {
                self.base_replay_throughput.store(self.radiance.load());
            }
            let reconnect_distance_criteria =
                shift_mapping.map(|sm| (self.prev_p - si.p).length() >= sm.min_dist);
            let prev_roughness_criteria =
                shift_mapping.map(|sm| self.prev_roughness >= sm.min_roughness);
            if let Some(sm) = &shift_mapping {
                let sm_vertex = sm.read_vertex();
                let is_last_vertex = self.depth == self.max_depth;
                let dist = (self.prev_p - si.p).length();
                let can_connect =
                    reconnect_distance_criteria.unwrap() & prev_roughness_criteria.unwrap();
                // if self.enable_debug {
                //     device_log!(
                //         "last hit light; d:{} can connect:{}",
                //         **self.depth,
                //         can_connect
                //     );
                // }
                // handle reconnection vertex hits surface light
                if (self.depth >= self.min_reconnect_depth) & can_connect {
                    if !sm_vertex.valid() & sm.is_base_path & is_last_vertex {
                        sm.write_vertex(ReconnectionVertex::from_comps_expr(
                            ReconnectionVertexComps {
                                direct: Expr::<FlatColor>::zeroed(),
                                indirect: Expr::<FlatColor>::zeroed(),
                                bary: si.bary,
                                direct_wi: Expr::<[f32; 3]>::zeroed(),
                                direct_light_pdf: 0.0f32.expr(),
                                wo: wo.into(),
                                inst_id: si.inst_id,
                                wi: Expr::<[f32; 3]>::zeroed(),
                                prim_id: si.prim_id,
                                prev_bsdf_pdf: **self.prev_bsdf_pdf,
                                bsdf_pdf: 0.0f32.expr(),
                                u_bsdf_select: 0.0f32.expr(),
                                dist,
                                depth: **self.depth,
                                type_: VertexType::LAST_HIT_LIGHT.expr(),
                            },
                        ));
                    } else if !sm.is_base_path & is_last_vertex {
                        // this line is incorrect
                        // not reversible
                        *rejected = true;
                        break;
                    }
                }
            }

            if self.depth >= self.max_depth {
                break;
            }
            *self.depth += 1;

            let direct_lighting = {
                let u_direct = sampler.next_3d();

                if !disable_nee {
                    self.sample_light(si, u_direct)
                } else {
                    DirectLighting::invalid(self.color_pipeline)
                }
            };
            let occluded = true.var();
            let u_bsdf = sampler.next_3d();
            let (bsdf_sample, features, direct) = if always_reconnect_first_hit {
                (
                    BsdfSample::invalid(self.color_pipeline.color_repr),
                    BsdfDenoisingFeatures {
                        albedo: Color::zero(self.color_pipeline.color_repr),
                        roughness: 0.0f32.expr(),
                    },
                    Color::zero(self.color_pipeline.color_repr),
                )
            } else {
                self.sample_surface_and_shade_direct(None, si, wo, direct_lighting, u_bsdf)
            };
            let u_select = u_bsdf.x;
            if self.depth == 1 {
                if let Some(df) = self.denoising {
                    df.first_hit_albedo.store(features.albedo);
                    *df.first_hit_roughness = features.roughness;
                }
            }
            let roughness = features.roughness;
            let roughness_criteria = shift_mapping.map(|sm| roughness >= sm.min_roughness);

            if !disable_nee & direct_lighting.valid {
                let shadow_ray = direct_lighting.shadow_ray;
                *occluded = self.scene.occlude(shadow_ray);
                if !occluded {
                    self.add_radiance(direct);
                }
                if self.depth == 1 {
                    self.base_replay_throughput.store(self.radiance.load());
                }
            }

            if let Some(sm) = &shift_mapping {
                let roughness_criteria = roughness_criteria.unwrap();
                let sm_vertex = sm.read_vertex();
                // perform reconnection
                if !sm.is_base_path & sm_vertex.valid() {
                    if self.depth > self.min_reconnect_depth {
                        let can_connect = reconnect_distance_criteria.unwrap()
                            & prev_roughness_criteria.unwrap()
                            & roughness_criteria;
                        // not reversible
                        if can_connect {
                            *rejected = true;
                            break;
                        }
                    }
                    // yes, we can connect
                    if sm_vertex.depth == self.depth {
                        let reconnection_vertex = sm_vertex;
                        let vertex_type = reconnection_vertex.type_;
                        let reconnect_si = self.scene.surface_interaction(
                            reconnection_vertex.inst_id,
                            reconnection_vertex.prim_id,
                            reconnection_vertex.bary,
                        );
                        let dist = (reconnect_si.p - si.p).length();
                        let wi = (reconnect_si.p - si.p).normalize();
                        let can_connect = (dist >= sm.min_dist) & roughness_criteria;
                        // failed to connect, reject
                        if !can_connect {
                            *rejected = true;
                            break;
                        }
                        let vis_ray = Ray::new_expr(
                            offset_ray_origin(si.p, face_forward(si.ng, wi)),
                            wi,
                            0.0,
                            dist * (1.0f32 - 1e-3f32),
                            Uint2::expr(si.inst_id, si.prim_id),
                            Uint2::expr(reconnect_si.inst_id, reconnect_si.prim_id),
                        );

                        let cos_theta_y2 = reconnect_si.ng.dot(wi).abs();
                        let cos_theta_x2 = reconnect_si
                            .ng
                            .dot(Expr::<Float3>::from(reconnection_vertex.wo))
                            .abs();
                        // prevent NaN
                        if cos_theta_y2 == 0.0 {
                            *rejected = true;
                            break;
                        }

                        let occluded = self.scene.occlude(vis_ray);
                        // visiblity check failed, reject
                        if occluded {
                            *rejected = true;
                            break;
                        }

                        /*
                         *
                         * x_i                 x_{i+1}                               x_{i+2}
                         *    vertex.prev_pdf          pdf(x_i -> x_{i+1}, x_{i+1} -> x_{i+2}) = pdf_x2
                         * y_i                 y_{i+1}                               y_{i+2} =x_{i+2}
                         *    pdf_y1                 pdf(x_i -> x_{i+1}, x_{i+1} -> x_{i+2}) = pdf_y2
                         */
                        let (f1, pdf_y1) = {
                            let f1_v = ColorVar::zero(self.color_pipeline.color_repr);
                            let pdf_y1_v = 0.0f32.var();
                            outline(|| {
                                let e = self.dispatch_surface(
                                    None,
                                    si.surface,
                                    si,
                                    **self.swl,
                                    &|closure| {
                                        closure.evaluate(wo, wi, **self.swl, &self.eval_context().1)
                                    },
                                );
                                f1_v.store(e.0);
                                *pdf_y1_v = e.1;
                            });
                            (f1_v.load(), pdf_y1_v.load())
                        };
                        let direct_wi = Expr::<Float3>::from(reconnection_vertex.direct_wi);
                        let (direct_f, roughness_y, f2, pdf_y2) = {
                            let direct_f_v = ColorVar::zero(self.color_pipeline.color_repr);
                            let roughness_y_v = 0.0f32.var();
                            let f2_v = ColorVar::zero(self.color_pipeline.color_repr);
                            let pdf_y2_v = 0.0f32.var();
                            outline(|| {
                                let e = self.dispatch_surface(
                                    None,
                                    reconnect_si.surface,
                                    reconnect_si,
                                    **self.swl,
                                    &|closure| {
                                        let (roughness_y, f2, pdf_y2) =
                                            if vertex_type == VertexType::LAST_HIT_LIGHT {
                                                // zero
                                                (
                                                    0.0f32.expr(),
                                                    Color::zero(self.color_pipeline.color_repr),
                                                    0.0f32.expr(),
                                                )
                                            } else {
                                                let (f, pdf) = closure.evaluate(
                                                    -wi,
                                                    Expr::<Float3>::from(reconnection_vertex.wi),
                                                    **self.swl,
                                                    &self.eval_context().1,
                                                );
                                                let roughness = closure.roughness(
                                                    -wi,
                                                    reconnection_vertex.u_bsdf_select,
                                                    **self.swl,
                                                    &self.eval_context().1,
                                                );
                                                (roughness, f, pdf)
                                            };
                                        let direct_f = if (direct_wi != 0.0).any() {
                                            let (f, bsdf_pdf) = closure.evaluate(
                                                -wi,
                                                direct_wi,
                                                **self.swl,
                                                &self.eval_context().1,
                                            );
                                            let w = mis_weight(
                                                reconnection_vertex.direct_light_pdf,
                                                bsdf_pdf,
                                                1,
                                            );
                                            f * w
                                        } else {
                                            Color::zero(self.color_pipeline.color_repr)
                                        };
                                        (direct_f, roughness_y, f2, pdf_y2)
                                    },
                                );
                                direct_f_v.store(e.0);
                                *roughness_y_v = e.1;
                                f2_v.store(e.2);
                                *pdf_y2_v = e.3;
                            });
                            (
                                direct_f_v.load(),
                                roughness_y_v.load(),
                                f2_v.load(),
                                pdf_y2_v.load(),
                            )
                        };

                        // check reversibility
                        // for path to be reversible, the newly formed path should also satisfy the roughness criteria
                        // if it fails, reject
                        if (vertex_type != VertexType::LAST_HIT_LIGHT)
                            & (roughness_y < sm.min_roughness)
                        {
                            *rejected = true;
                            break;
                        }

                        // let pdf_x = if vertex_type != VertexType::INTERIOR {
                        //     reconnection_vertex.prev_bsdf_pdf
                        // } else {
                        //     reconnection_vertex.prev_bsdf_pdf * reconnection_vertex.bsdf_pdf
                        // };
                        // let pdf_y = pdf_y1 * pdf_y2;
                        let compute_ratio = |pdf_y: Expr<f32>, pdf_x: Expr<f32>| {
                            if pdf_x == 0.0 {
                                (pdf_y == 0.0).select(1.0f32.expr(), 0.0f32.expr())
                            } else {
                                pdf_y / pdf_x
                            }
                        };
                        let pdf_ratio = (pdf_y1 / reconnection_vertex.prev_bsdf_pdf).var();
                        if vertex_type != VertexType::LAST_HIT_LIGHT {
                            *pdf_ratio *= compute_ratio(pdf_y2, reconnection_vertex.bsdf_pdf);
                        }
                        if pdf_ratio <= 0.0 {
                            *rejected = true;
                            break;
                        }
                        let throughput = {
                            let light_ctx = self.eval_context().0;
                            let reconnect_instance = self
                                .scene
                                .meshes
                                .mesh_instances()
                                .read(reconnect_si.inst_id);
                            let le = if reconnect_instance.light.valid() {
                                self.scene
                                    .lights
                                    .le(vis_ray, reconnect_si, **self.swl, &light_ctx)
                            } else {
                                Color::zero(self.color_pipeline.color_repr)
                            };
                            let light_pdf = if reconnect_instance.light.valid() {
                                self.scene.lights.pdf_direct(
                                    reconnect_si,
                                    PointNormal::new_expr(si.p, si.ng),
                                    &light_ctx,
                                )
                            } else {
                                0.0f32.expr()
                            };

                            let w = if self.use_nee
                                & (!self.shift_mapping_no_nee | (self.depth == 1))
                            {
                                mis_weight(pdf_y1, light_pdf, 1)
                            } else {
                                1.0f32.expr()
                            };
                            let vertex_le = le * w;
                            let vertex_le = if self.indirect_only & (self.depth == 1) {
                                Color::zero(self.color_pipeline.color_repr)
                            } else {
                                vertex_le
                            };

                            let f_pdf = f1 / pdf_y1;
                            let (_, cont_prob) = self.compute_contibue_prob(
                                reconnection_vertex.depth,
                                self.reconnect_beta.load() * f_pdf,
                            );
                            f_pdf
                                * (vertex_le
                                    + direct_f
                                        * Color::from_flat(
                                            self.color_pipeline.color_repr,
                                            reconnection_vertex.direct,
                                        )
                                    + if pdf_y2 > 0.0f32 {
                                        f2 * Color::from_flat(
                                            self.color_pipeline.color_repr,
                                            reconnection_vertex.indirect,
                                        ) / pdf_y2
                                    } else {
                                        Color::zero(self.color_pipeline.color_repr)
                                    })
                                / cont_prob
                            // is this correct???
                        };

                        self.add_radiance(throughput);

                        let jacobian = pdf_ratio
                            * (cos_theta_y2 / cos_theta_x2).abs()
                            * (reconnection_vertex.dist / dist).sqr();
                        let jacobian = jacobian.is_finite().select(jacobian, 0.0f32.expr());
                        *sm.success = jacobian > 0.0;
                        *sm.jacobian = jacobian;
                        if !sm.success {
                            *rejected = true;
                        }
                        break;
                    }
                }
            }
            if always_reconnect_first_hit {
                break;
            }
            let f = &bsdf_sample.color;
            // if debug_mode() {
            //     lc_assert!(f.min().ge(0.0));
            // };

            self.mul_beta(f / bsdf_sample.pdf);
            if let Some(sm) = &shift_mapping {
                if self.depth > self.min_reconnect_depth {
                    let reconnect_vertex = sm.read_vertex();
                    let dist = (self.prev_p - si.p).length();
                    let can_connect = reconnect_distance_criteria.unwrap()
                        & prev_roughness_criteria.unwrap()
                        & roughness_criteria.unwrap();
                    // if self.enable_debug {
                    //     device_log!("LAST_NEE; d:{} can connect:{}", **self.depth, can_connect);
                    // }
                    if !reconnect_vertex.valid() & sm.is_base_path & can_connect {
                        sm.write_vertex(ReconnectionVertex::from_comps_expr(
                            ReconnectionVertexComps {
                                direct: if direct_lighting.valid & !occluded {
                                    direct_lighting.irradiance.flatten() / direct_lighting.pdf
                                } else {
                                    Expr::<FlatColor>::zeroed()
                                },
                                indirect: Expr::<FlatColor>::zeroed(),
                                bary: si.bary,
                                direct_wi: direct_lighting.wi.into(),
                                direct_light_pdf: direct_lighting.pdf,
                                wo: wo.into(),
                                inst_id: si.inst_id,
                                wi: bsdf_sample.wi.into(),
                                prim_id: si.prim_id,
                                prev_bsdf_pdf: **self.prev_bsdf_pdf,
                                bsdf_pdf: bsdf_sample.pdf,
                                u_bsdf_select: u_select,
                                dist,
                                depth: self.depth - 1,
                                type_: VertexType::LAST_NEE.expr(),
                            },
                        ));
                        self.reconnect_beta
                            .store(Color::one(self.color_pipeline.color_repr));
                        self.reconnect_radiance
                            .store(Color::zero(self.color_pipeline.color_repr));
                    }
                    // we found a connectible vertex, this means the shift is not reversible
                    // this is because if the shift is reversible, we should already connect to the
                    // vertex at a smaller depth
                    if !sm.is_base_path & can_connect {
                        *rejected = true;
                        break;
                    }
                }
            }
            if (bsdf_sample.pdf <= 0.0) | !bsdf_sample.valid | (bsdf_sample.color.reduce_min() < 0.0) {
                // if self.enable_debug {
                //     device_log!(
                //         "bsdf sample invalid, d:{} {} {}",
                //         **self.depth,
                //         bsdf_sample.pdf,
                //         bsdf_sample.wi
                //     );
                // }
                break;
            }
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
                *self.prev_p = si.p;
                *self.prev_roughness = roughness;
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
        // if self.enable_debug {
        //     device_log!("end, d:{}", **self.depth);
        // }

        if self.clamp_indirect > 0.0 {
            let indirect = self.radiance.load() - self.base_replay_throughput.load();
            let indirect = indirect.clamp(self.clamp_indirect.expr());
            self.radiance
                .store(self.base_replay_throughput.load() + indirect);
        }

        if let Some(sm) = &shift_mapping {
            let reconnect_vertex = sm.read_vertex().var();
            if reconnect_vertex.valid()
                & (reconnect_vertex.type_ != VertexType::LAST_HIT_LIGHT)
                & sm.is_base_path
            {
                *reconnect_vertex.indirect = self.reconnect_radiance.load().flatten();
                sm.write_vertex(reconnect_vertex);
            }
            if !sm.is_base_path {
                if reconnect_vertex.valid() {
                    if **sm.success {
                        lc_assert!(sm.jacobian.gt(0.0));
                    } else {
                        lc_assert!(sm.jacobian.eq(0.0));
                    }
                } else {
                    *sm.success = !rejected;
                    *sm.jacobian = sm.success.select(1.0f32.expr(), 0.0f32.expr());
                }
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
#[serde(crate = "serde")]
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
    pub debug_depth: Option<u32>,
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
            debug_depth: None,
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
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct ReconnectionVertex {
    pub direct: FlatColor,
    pub indirect: FlatColor,
    pub bary: Float2,
    pub direct_wi: [f32; 3],
    pub direct_light_pdf: f32,
    pub wo: [f32; 3],
    pub inst_id: u32,
    pub wi: [f32; 3],
    pub prim_id: u32,
    pub prev_bsdf_pdf: f32,
    pub bsdf_pdf: f32,
    pub u_bsdf_select: f32,
    pub dist: f32,
    pub depth: u32,
    pub type_: u32,
}
impl ReconnectionVertexExpr {
    pub fn valid(&self) -> Expr<bool> {
        self.type_.ne(VertexType::INVALID)
    }
}

pub struct ReconnectionShiftMapping<'a> {
    pub min_dist: Expr<f32>,
    pub min_roughness: Expr<f32>,
    pub is_base_path: Var<bool>,
    // pub vertex: Var<ReconnectionVertex>,
    pub read_vertex: Box<dyn Fn() -> Expr<ReconnectionVertex> + 'a>,
    pub write_vertex: Box<dyn Fn(Expr<ReconnectionVertex>) + 'a>,
    pub jacobian: Var<f32>,
    pub success: Var<bool>,
}
impl<'a> ReconnectionShiftMapping<'a> {
    pub fn read_vertex(&self) -> Expr<ReconnectionVertex> {
        (self.read_vertex)()
    }
    pub fn write_vertex(&self, x: impl AsExpr<Value = ReconnectionVertex>) {
        (self.write_vertex)(x.as_expr())
    }
}
#[derive(Clone, Copy, Value, Debug)]
#[luisa(crate = "luisa")]
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
        pt.debug_depth = self.config.debug_depth.map(|x| x.expr());
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
                        let (ray, ray_w) = scene.camera.generate_ray(
                            &scene,
                            film.filter(),
                            shifted,
                            sampler,
                            color_pipeline.color_repr,
                            swl,
                        );
                        let swl = swl.var();
                        let l = self.radiance(&scene, color_pipeline, ray, swl, sampler);
                        film.add_sample(p.cast_f32(), &l, **swl, ray_w);
                    });
                }
            ));
        log::info!(
            "Render kernel has {} arguments, {} captures!",
            kernel.num_arguments(),
            kernel.num_capture_arguments()
        );
        let mut cnt = 0;
        let progress = util::create_progess_bar(self.spp as usize, "spp");
        let mut acc_time = 0.0;
        let mut stats: RenderStats = Default::default();
        let output_image: Tex2d<Float4> = self.device.create_tex2d(
            PixelStorage::Float4,
            scene.camera.resolution().x,
            scene.camera.resolution().y,
            1,
        );
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
            cnt += cur_pass;
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
            progress.inc(cur_pass as u64);
        }
        if session.save_stats {
            let file = File::create(format!("{}.json", session.name)).unwrap();
            let json = serde_json::to_value(&stats).unwrap();
            let writer = BufWriter::new(file);
            serde_json::to_writer(writer, &json).unwrap();
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
