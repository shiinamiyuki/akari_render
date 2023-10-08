use std::marker::PhantomData;
use std::sync::Arc;

use luisa::rtx::{CommittedHit, Hit};

use crate::color::{Color, ColorPipeline, ColorRepr, FlatColor, SampledWavelengths};

use crate::light::{LightAggregate, LightEvalContext};

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

impl Scene {
    pub fn si_from_hitinfo(
        &self,
        inst_id: Expr<u32>,
        prim_id: Expr<u32>,
        bary: Expr<Float2>,
    ) -> SurfaceInteraction {
        let shading_triangle = self.meshes.shading_triangle(inst_id, prim_id);
        let p = shading_triangle.p(bary);
        let uv = shading_triangle.uv(bary);
        let frame = shading_triangle.ortho_frame(bary);
        SurfaceInteraction {
            inst_id,
            prim_id,
            bary,
            ng: shading_triangle.ng,
            p,
            uv,
            frame,
            valid: true.expr(),
            surface: shading_triangle.surface,
        }
    }
    #[tracked]
    pub fn _trace_closest(&self, ray: Expr<Ray>) -> Expr<Hit> {
        let ro: Expr<[f32; 3]> = ray.o.into();
        let rd: Expr<[f32; 3]> = ray.d.into();
        let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
        self.meshes
            .accel
            .var()
            .trace_closest_masked(rtx_ray, 255u32.expr())
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
    pub fn intersect(&self, ray: Expr<Ray>) -> SurfaceInteraction {
        if !self.use_rq {
            let hit = self._trace_closest(ray);
            if !hit.miss() {
                let inst_id = hit.inst_id;
                let prim_id = hit.prim_id;
                let bary = Float2::expr(hit.u, hit.v);
                self.si_from_hitinfo(inst_id, prim_id, bary)
            } else {
                SurfaceInteraction::invalid()
            }
        } else {
            let hit = self._trace_closest_rq(ray);
            if hit.triangle_hit() {
                let inst_id = hit.inst_id;
                let prim_id = hit.prim_id;
                let bary = hit.bary;
                self.si_from_hitinfo(inst_id, prim_id, bary)
            } else {
                SurfaceInteraction::invalid()
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
