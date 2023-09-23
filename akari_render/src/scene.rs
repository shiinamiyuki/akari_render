use std::marker::PhantomData;
use std::sync::Arc;

use crate::color::{Color, ColorPipeline, ColorRepr, FlatColor, SampledWavelengths};

use crate::light::{
    FlatLightSample, FlatLightSampleExpr, LightAggregate, LightEvalContext, LightEvaluator,
};

use crate::svm::surface::{
    BsdfEvalContext, FlatBsdfSample, FlatBsdfSampleComps, FlatSurfaceEvalResult,
    FlatSurfaceEvalResultComps, Surface, SurfaceEvaluator, SURFACE_EVAL_ALBEDO, SURFACE_EVAL_COLOR,
    SURFACE_EVAL_EMISSION, SURFACE_EVAL_PDF, SURFACE_EVAL_ROUGHNESS,
};
use crate::svm::{ShaderRef, Svm};
use crate::{camera::Camera, geometry::*, interaction::*, mesh::*, *};

pub struct Scene {
    pub svm: Arc<Svm>,
    pub lights: LightAggregate,
    pub meshes: Arc<MeshAggregate>,
    pub camera: Arc<dyn Camera>,
    pub device: Device,
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
    pub fn light_evaluator(
        &self,
        color_pipeline: ColorPipeline,
        surface_eval: &SurfaceEvaluator,
        ad_mode: ADMode,
    ) -> LightEvaluator {
        let le = {
            self.device.create_callable::<fn(
                Expr<Ray>,
                Expr<SurfaceInteraction>,
                Expr<SampledWavelengths>,
            ) -> Expr<FlatColor>>(
                &|ray, si: Expr<SurfaceInteraction>, swl| {
                    let inst_id = si.inst_id();
                    let instance = self.meshes.mesh_instances.var().read(inst_id);
                    if_!(instance.light().valid(), {
                        let ctx = LightEvalContext {
                            meshes: &self.meshes,
                            color_pipeline,
                            surface_eval
                        };
                        self.lights.le(ray, si,swl, &ctx).flatten()
                    }, else {
                      Color::zero(color_pipeline.color_repr).flatten()
                    })
                },
            )
        };
        let sample = {
            self.device.create_callable::<fn(
                Expr<PointNormal>,
                Expr<Float3>,
                Expr<SampledWavelengths>,
            ) -> Expr<FlatLightSample>>(&|pn, u, swl| {
                let ctx = LightEvalContext {
                    meshes: &self.meshes,
                    color_pipeline,
                    surface_eval,
                };
                let sample = self.lights.sample_direct(pn, u.x, u.yz(), swl, &ctx);
                FlatLightSampleExpr::new(
                    sample.li.flatten(),
                    sample.pdf,
                    sample.wi,
                    sample.shadow_ray,
                    sample.n,
                )
            })
        };
        let pdf = {
            let scene = self.clone();
            self.device.create_callable::<fn(
                Expr<SurfaceInteraction>,
                Expr<PointNormal>,
                Expr<SampledWavelengths>,
            ) -> Expr<f32>>(&|si, pn, _swl| {
                let inst_id = si.inst_id();
                let instance = scene.meshes.mesh_instances.var().read(inst_id);
                if_!(instance.light().valid(), {
                    let ctx = LightEvalContext {
                        meshes: &self.meshes,
                        color_pipeline,
                        surface_eval
                    };
                    scene.lights.pdf_direct(si, pn, &ctx)
                }, else {
                    0.0f32.expr()
                })
            })
        };
        LightEvaluator {
            color_pipeline,
            le,
            sample,
            pdf,
        }
    }
    pub fn surface_evaluator(
        &self,
        color_pipeline: ColorPipeline,
        ad_mode: ADMode,
    ) -> SurfaceEvaluator {
        let eval = {
            self.device.create_callable::<fn(
                Expr<ShaderRef>,
                Expr<SurfaceInteraction>,
                Expr<Float3>,
                Expr<Float3>,
                Expr<SampledWavelengths>,
                Expr<u32>,
            ) -> Expr<FlatSurfaceEvalResult>>(
                &|shader_ref: Expr<ShaderRef>,
                  si: Expr<SurfaceInteraction>,
                  wo: Expr<Float3>,
                  wi: Expr<Float3>,
                  swl: Expr<SampledWavelengths>,
                  mode: Expr<u32>| {
                    let ctx = BsdfEvalContext {
                        color_repr: color_pipeline.color_repr,
                        _marker: PhantomData,
                        ad_mode,
                    };
                    let color_repr = ctx.color_repr;
                    let (color, pdf, albedo, emission, roughness) =
                        self.svm
                            .dispatch_surface(shader_ref, color_pipeline, si, swl, |closure| {
                                let color = if_!(
                                    (mode & SURFACE_EVAL_COLOR).ne(0),
                                    { closure.evaluate(wo, wi, swl, &ctx) },
                                    else,
                                    { Color::zero(color_repr) }
                                );
                                let pdf = if_!(
                                    (mode & SURFACE_EVAL_PDF).ne(0),
                                    { closure.pdf(wo, wi, swl, &ctx) },
                                    else,
                                    { 0.0.into() }
                                );
                                let albedo = if_!(
                                    (mode & SURFACE_EVAL_ALBEDO).ne(0),
                                    { closure.albedo(wo, swl, &ctx) },
                                    else,
                                    { Color::zero(color_repr) }
                                );
                                let emission = if_!(
                                    (mode & SURFACE_EVAL_EMISSION).ne(0),
                                    { closure.emission(wo, swl, &ctx) },
                                    else,
                                    { Color::zero(color_repr) }
                                );
                                let roughness = if_!(
                                    (mode & SURFACE_EVAL_ROUGHNESS).ne(0),
                                    { closure.roughness(wo, swl, &ctx) },
                                    else,
                                    { 0.0.into() }
                                );
                                (color, pdf, albedo, emission, roughness)
                            });
                    Expr::<FlatSurfaceEvalResult>::from_comps_expr(FlatSurfaceEvalResultComps {
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
            self.device.create_callable::<fn(
                Expr<ShaderRef>,
                Expr<SurfaceInteraction>,
                Expr<Float3>,
                Expr<Float3>,
                Var<SampledWavelengths>,
            ) -> Expr<FlatBsdfSample>>(
                &|shader_ref: Expr<ShaderRef>,
                  si: Expr<SurfaceInteraction>,
                  wo: Expr<Float3>,
                  u: Expr<Float3>,
                  swl: Var<SampledWavelengths>| {
                    let ctx = BsdfEvalContext {
                        color_repr: color_pipeline.color_repr,
                        _marker: PhantomData,
                        ad_mode,
                    };
                    let sample = self.svm.dispatch_surface(
                        shader_ref,
                        color_pipeline,
                        si,
                        *swl,
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
    // pub fn load_from_path<P: AsRef<Path>>(device: Device, path: P) -> Arc<Self> {
    //     let path = PathBuf::from(path.as_ref());
    //     let canonical = fs::canonicalize(&path).unwrap();
    //     let parent_path = canonical.parent().unwrap();
    //     let serialized = std::fs::read_to_string(path).unwrap();
    //     Self::load_from_str(
    //         device,
    //         &serialized,
    //         Arc::new(LocalFileResolver::new(vec![PathBuf::from(parent_path)])),
    //     )
    // }
    // pub fn load_from_str(
    //     device: Device,
    //     desc: &str,
    //     file_resolver: Arc<dyn FileResolver + Send + Sync>,
    // ) -> Arc<Self> {
    //     let graph: node::Scene = serde_json::from_str(desc).unwrap_or_else(|e| {
    //         log::error!("error during scene loading:{:}", e);
    //         std::process::exit(-1);
    //     });
    //     let loader = SceneLoader::new(device, Rc::new(graph), file_resolver);
    //     loader.load()
    // }
    // pub fn env_map(&self, w: Expr<Float3>, evals: &Evaluators) -> Color {
    //     // TODO: fix this
    //     let (theta, phi) = xyz_to_spherical(w);
    //     let u = phi / (2.0 * PI);
    //     let v = theta / PI;
    //     let si = var!(SurfaceInteraction);
    //     si.set_geometry(zeroed::<SurfaceLocalGeometry>().set_uv(Float2::expr(u, v)));
    //     evals
    //         .texture
    //         .evaluate_color(self.env_map.var().read(0), si.load())
    // }
    pub fn si_from_hitinfo(
        &self,
        inst_id: Expr<u32>,
        prim_id: Expr<u32>,
        bary: Expr<Float2>,
    ) -> SurfaceInteractionExpr {
        let shading_triangle = self.meshes.shading_triangle(inst_id, prim_id);
        let p = shading_triangle.p(bary);
        let n = shading_triangle.n(bary);
        let uv = shading_triangle.uv(bary);
        let tt = shading_triangle.tangent(bary);
        let ss = shading_triangle.bitangent(bary);
        let geometry = SurfaceLocalGeometry::from_comps_expr(SurfaceLocalGeometryComps {
            p,
            ng: shading_triangle.ng,
            ns: n,
            uv,
            tangent: shading_triangle.tangent(bary),
            bitangent: shading_triangle.bitangent(bary),
        });
        SurfaceInteractionExpr::new(
            inst_id,
            prim_id,
            bary,
            geometry,
            FrameExpr::new(n, tt, ss),
            true.expr(),
        )
    }
    #[tracked]
    pub fn intersect(&self, ray: Expr<Ray>) -> Expr<SurfaceInteraction> {
        let ro = ray.o;
        let rd = ray.d;
        let rtx_ray = rtx::RayExpr::new(ro, ray.t_min, rd, ray.t_max);

        let hit = self.meshes.accel.var().query_all(
            rtx_ray,
            255,
            rtx::RayQuery {
                on_triangle_hit: |candidate: rtx::TriangleCandidate| {
                    if_!(
                        (candidate.inst().ne(ray.exclude0().x)
                            | candidate.prim().ne(ray.exclude0().y))
                            & (candidate.inst().ne(ray.exclude1().x)
                                | candidate.prim().ne(ray.exclude1().y)),
                        {
                            candidate.commit();
                        }
                    );
                },
                on_procedural_hit: |_| {},
            },
        );

        // cpu_dbg!(hit);
        if hit.triangle_hit() {
            let inst_id = hit.inst_id();
            let prim_id = hit.prim_id();
            let bary = hit.bary();
            self.si_from_hitinfo(inst_id, prim_id, bary)
        } else {
            let si = Var::<SurfaceInteraction>::zeroed();
            si.valid = false.expr();
            si
        }
    }
    #[tracked]
    pub fn occlude(&self, ray: Expr<Ray>) -> Expr<bool> {
        let ro = ray.o;
        let rd = ray.d;
        let rtx_ray = rtx::RayExpr::new(ro, ray.t_min, rd, ray.t_max);

        let hit = self.meshes.accel.var().query_any(
            rtx_ray,
            255,
            rtx::RayQuery {
                on_triangle_hit: |candidate: rtx::TriangleCandidate| {
                    if_!(
                        (candidate.inst.ne(ray.exclude0.x) | candidate.prim.ne(ray.exclude0.y))
                            & (candidate.inst.ne(ray.exclude1.x)
                                | candidate.prim.ne(ray.exclude1.y)),
                        {
                            candidate.commit();
                        }
                    );
                },
                on_procedural_hit: |_| {},
            },
        );
        // cpu_dbg!(ray);
        // cpu_dbg!(hit);
        !hit.miss()
    }
}

