use std::marker::PhantomData;
use std::sync::Arc;

use luisa::rtx::{CommittedHit, Hit};

use crate::color::{Color, ColorPipeline, ColorRepr, FlatColor, SampledWavelengths};

use crate::light::{
    FlatLightSample, FlatLightSampleExpr, LightAggregate, LightEvalContext, LightEvaluator,
};

use crate::svm::surface::{
    BsdfEvalContext, EvalSurfaceCallable, FlatBsdfSample, FlatBsdfSampleComps,
    FlatSurfaceEvalResult, FlatSurfaceEvalResultComps, SampleSurfaceCallable, Surface,
    SurfaceEvaluator, SURFACE_EVAL_ALBEDO, SURFACE_EVAL_COLOR, SURFACE_EVAL_EMISSION,
    SURFACE_EVAL_PDF, SURFACE_EVAL_ROUGHNESS,
};
use crate::svm::{ShaderRef, Svm};
use crate::{camera::Camera, geometry::*, interaction::*, mesh::*, *};

pub struct Scene {
    pub svm: Arc<Svm>,
    pub lights: LightAggregate,
    pub meshes: Arc<MeshAggregate>,
    pub camera: Arc<dyn Camera>,
    pub device: Device,
    pub use_rq: bool,
    pub printer: Printer,
    // pub env_map: Buffer<TagIndex>,
}
pub struct Evaluators {
    pub color_pipeline: ColorPipeline,
    pub surface: SurfaceEvaluator,
    pub light: LightEvaluator,
}
impl Evaluators {
    pub fn color_repr(&self) -> ColorRepr {
        self.color_pipeline.color_repr
    }
}

impl Scene {
    pub fn evaluators(&self, color_pipeline: ColorPipeline, ad_mode: ADMode) -> Evaluators {
        let surface = self.surface_evaluator(color_pipeline, ad_mode);
        let light = self.light_evaluator(color_pipeline, &surface, ad_mode);
        Evaluators {
            color_pipeline,
            light,
            surface,
        }
    }
    #[tracked]
    pub fn light_evaluator(
        &self,
        color_pipeline: ColorPipeline,
        surface_eval: &SurfaceEvaluator,
        ad_mode: ADMode,
    ) -> LightEvaluator {
        let le = {
            Callable::<
                fn(
                    Expr<Ray>,
                    Expr<SurfaceInteraction>,
                    Expr<SampledWavelengths>,
                ) -> Expr<FlatColor>,
            >::new(&self.device, &|ray, si: Expr<SurfaceInteraction>, swl| {
                let inst_id = si.inst_id;
                let instance = self.meshes.mesh_instances.var().read(inst_id);
                if instance.light.valid() {
                    let ctx = LightEvalContext {
                        meshes: &self.meshes,
                        color_pipeline,
                        surface_eval,
                    };
                    self.lights.le(ray, si, swl, &ctx).flatten()
                } else {
                    Color::zero(color_pipeline.color_repr).flatten()
                }
            })
        };
        let sample = {
            Callable::<
                fn(
                    Expr<PointNormal>,
                    Expr<Float3>,
                    Expr<SampledWavelengths>,
                ) -> Expr<FlatLightSample>,
            >::new(&self.device, |pn, u, swl| {
                let ctx = LightEvalContext {
                    meshes: &self.meshes,
                    color_pipeline,
                    surface_eval,
                };
                let sample = self.lights.sample_direct(pn, u.x, u.yz(), swl, &ctx);
                FlatLightSample::new_expr(
                    sample.li.flatten(),
                    sample.pdf,
                    sample.wi,
                    sample.shadow_ray,
                    sample.n,
                    sample.valid,
                )
            })
        };
        let pdf = {
            let scene = self.clone();
            Callable::<
                fn(
                    Expr<SurfaceInteraction>,
                    Expr<PointNormal>,
                    Expr<SampledWavelengths>,
                ) -> Expr<f32>,
            >::new(&self.device, |si, pn, _swl| {
                let inst_id = si.inst_id;
                let instance = scene.meshes.mesh_instances.var().read(inst_id);
                if instance.light.valid() {
                    let ctx = LightEvalContext {
                        meshes: &self.meshes,
                        color_pipeline,
                        surface_eval,
                    };
                    scene.lights.pdf_direct(si, pn, &ctx)
                } else {
                    0.0f32.expr()
                }
            })
        };
        LightEvaluator {
            color_pipeline,
            le,
            sample,
            pdf,
        }
    }
    #[tracked]
    pub fn surface_evaluator(
        &self,
        color_pipeline: ColorPipeline,
        ad_mode: ADMode,
    ) -> SurfaceEvaluator {
        let eval = {
            EvalSurfaceCallable::new(
                &self.device,
                |shader_ref: Expr<ShaderRef>,
                 si: Expr<SurfaceInteraction>,
                 wo: Expr<Float3>,
                 wi: Expr<Float3>,
                 swl: Expr<SampledWavelengths>,
                 mode: Expr<u32>| {
                    let check_wi = || {
                        if debug_mode() {
                            lc_assert!(wi.is_finite().all());
                            lc_assert!(wi.ne(0.0).any());
                        }
                    };
                    let check_wo = || {
                        if debug_mode() {
                            lc_assert!(wo.is_finite().all());
                            lc_assert!(wo.ne(0.0).any());
                        }
                    };
                    check_wo();
                    let ctx = BsdfEvalContext {
                        color_repr: color_pipeline.color_repr,
                        _marker: PhantomData,
                        ad_mode,
                    };
                    let color_repr = ctx.color_repr;
                    let (color, pdf, albedo, emission, roughness) =
                        self.svm
                            .dispatch_surface(shader_ref, color_pipeline, si, swl, |closure| {
                                let color = if (mode & SURFACE_EVAL_COLOR).ne(0) {
                                    check_wi();
                                    closure.evaluate(wo, wi, swl, &ctx)
                                } else {
                                    Color::zero(color_repr)
                                };
                                let pdf = if (mode & SURFACE_EVAL_PDF).ne(0) {
                                    check_wi();
                                    closure.pdf(wo, wi, swl, &ctx)
                                } else {
                                    0.0.expr()
                                };
                                let albedo = if (mode & SURFACE_EVAL_ALBEDO).ne(0) {
                                    closure.albedo(wo, swl, &ctx)
                                } else {
                                    Color::zero(color_repr)
                                };
                                let emission = if (mode & SURFACE_EVAL_EMISSION).ne(0) {
                                    closure.emission(wo, swl, &ctx)
                                } else {
                                    Color::zero(color_repr)
                                };

                                let roughness = if (mode & SURFACE_EVAL_ROUGHNESS).ne(0) {
                                    closure.roughness(wo, swl, &ctx)
                                } else {
                                    0.0.expr()
                                };
                                (color, pdf, albedo, emission, roughness)
                            });
                    FlatSurfaceEvalResult::from_comps_expr(FlatSurfaceEvalResultComps {
                        color: color.flatten(),
                        pdf,
                        albedo: albedo.flatten(),
                        emission: emission.flatten(),
                        roughness,
                    })
                },
            )
        };
        let sample = {
            SampleSurfaceCallable::new(
                &self.device,
                |shader_ref: Expr<ShaderRef>,
                 si: Expr<SurfaceInteraction>,
                 wo: Expr<Float3>,
                 u: Expr<Float3>,
                 swl: Var<SampledWavelengths>| {
                    if debug_mode() {
                        lc_assert!(u.is_finite().all());
                        lc_assert!(wo.is_finite().all());
                        lc_assert!(u.ne(0.0).any());
                        lc_assert!(wo.ne(0.0).any());
                    }
                    let ctx = BsdfEvalContext {
                        color_repr: color_pipeline.color_repr,
                        _marker: PhantomData,
                        ad_mode,
                    };
                    let sample = self.svm.dispatch_surface(
                        shader_ref,
                        color_pipeline,
                        si,
                        **swl,
                        |closure| closure.sample(wo, u.x, u.yz(), swl, &ctx),
                    );
                    FlatBsdfSample::from_comps_expr(FlatBsdfSampleComps {
                        wi: sample.wi,
                        pdf: sample.pdf,
                        valid: sample.valid,
                        color: sample.color.flatten(),
                    })
                },
            )
        };
        SurfaceEvaluator {
            color_repr: color_pipeline.color_repr,
            eval,
            sample,
        }
    }

    pub fn si_from_hitinfo(
        &self,
        inst_id: Expr<u32>,
        prim_id: Expr<u32>,
        bary: Expr<Float2>,
    ) -> Expr<SurfaceInteraction> {
        let shading_triangle = self.meshes.shading_triangle(inst_id, prim_id);
        let p = shading_triangle.p(bary);
        let uv = shading_triangle.uv(bary);
        let frame = shading_triangle.ortho_frame(bary);
        SurfaceInteraction::from_comps_expr(SurfaceInteractionComps {
            inst_id,
            prim_id,
            bary,
            ng: shading_triangle.ng,
            p,
            uv,
            frame,
            valid: true.expr(),
        })
    }
    #[tracked]
    pub fn _trace_closest(&self, ray: Expr<Ray>) -> Expr<Hit> {
        let ro: Expr<[f32; 3]> = ray.o.into();
        let rd: Expr<[f32; 3]> = ray.d.into();
        let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
        self.meshes.accel.var().trace_closest_masked(rtx_ray, 255u32.expr())
    }
    #[tracked]
    pub fn _trace_closest_rq(&self, ray: Expr<Ray>) -> Expr<CommittedHit> {
        let ro: Expr<[f32; 3]> = ray.o.into();
        let rd: Expr<[f32; 3]> = ray.d.into();
        let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
        self.meshes.accel.var().query_all(
            rtx_ray,
            u32::MAX,
            rtx::RayQuery {
                on_triangle_hit: |candidate: rtx::TriangleCandidate| {
                    if (candidate.inst.ne(ray.exclude0.x) | candidate.prim.ne(ray.exclude0.y))
                        & (candidate.inst.ne(ray.exclude1.x) | candidate.prim.ne(ray.exclude1.y))
                    {
                        candidate.commit();
                    }
                },
                on_procedural_hit: |_| {},
            },
        )
    }
    #[tracked]
    pub fn intersect(&self, ray: Expr<Ray>) -> Expr<SurfaceInteraction> {
        if !self.use_rq {
            let hit = self._trace_closest(ray);
            if !hit.miss() {
                let inst_id = hit.inst_id;
                let prim_id = hit.prim_id;
                let bary = Float2::expr(hit.u, hit.v);
                self.si_from_hitinfo(inst_id, prim_id, bary)
            } else {
                let si = Var::<SurfaceInteraction>::zeroed();
                *si.valid = false.expr();
                **si
            }
        } else {
            let hit = self._trace_closest_rq(ray);
            if hit.triangle_hit() {
                let inst_id = hit.inst_id;
                let prim_id = hit.prim_id;
                let bary = hit.bary;
                self.si_from_hitinfo(inst_id, prim_id, bary)
            } else {
                let si = Var::<SurfaceInteraction>::zeroed();
                *si.valid = false.expr();
                **si
            }
        }
    }
    #[tracked]
    pub fn occlude(&self, ray: Expr<Ray>) -> Expr<bool> {
        let ro: Expr<[f32; 3]> = ray.o.into();
        let rd: Expr<[f32; 3]> = ray.d.into();
        let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
        if !self.use_rq {
            self.meshes.accel.var().trace_any(rtx_ray)
        } else {
            let hit = self.meshes.accel.var().query_any(
                rtx_ray,
                u32::MAX,
                rtx::RayQuery {
                    on_triangle_hit: |candidate: rtx::TriangleCandidate| {
                        if (candidate.inst.ne(ray.exclude0.x) | candidate.prim.ne(ray.exclude0.y))
                            & (candidate.inst.ne(ray.exclude1.x)
                                | candidate.prim.ne(ray.exclude1.y))
                        {
                            candidate.commit();
                        }
                    },
                    on_procedural_hit: |_| {},
                },
            );
            !hit.miss()
        }
    }
}
