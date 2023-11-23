use luisa::rtx::{AccelTraceOptions, CommittedHit, SurfaceHit};
use std::sync::Arc;

use crate::heap::MegaHeap;
use crate::light::LightAggregate;

use crate::svm::Svm;
use crate::{camera::Camera, geometry::*, interaction::*, mesh::*, *};

pub struct Scene {
    pub svm: Arc<Svm>,
    pub lights: LightAggregate,
    pub meshes: Arc<MeshAggregate>,
    pub camera: Arc<dyn Camera>,
    pub device: Device,
    pub use_rq: bool,
    pub heap: Arc<MegaHeap>,
    // pub env_map: Buffer<TagIndex>,
}

impl Scene {
    pub fn trace_options(&self) -> AccelTraceOptions {
        AccelTraceOptions {
            mask: 0xff.expr(),
            ..Default::default()
        }
    }
    pub fn surface_interaction(
        &self,
        inst_id: Expr<u32>,
        prim_id: Expr<u32>,
        bary: Expr<Float2>,
    ) -> SurfaceInteraction {
        self.meshes.surface_interaction(inst_id, prim_id, bary)
    }
    #[tracked(crate = "luisa")]
    pub fn _trace_closest(&self, ray: Expr<Ray>) -> Expr<SurfaceHit> {
        let ro: Expr<[f32; 3]> = ray.o.into();
        let rd: Expr<[f32; 3]> = ray.d.into();
        let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
        self.meshes.accel.intersect(rtx_ray, self.trace_options())
    }
    #[tracked(crate = "luisa")]
    pub fn _trace_closest_rq(&self, ray: Expr<Ray>) -> Expr<CommittedHit> {
        let ro: Expr<[f32; 3]> = ray.o.into();
        let rd: Expr<[f32; 3]> = ray.d.into();
        let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
        self.meshes
            .accel
            .traverse(rtx_ray, self.trace_options())
            .on_surface_hit(|candidate: rtx::SurfaceCandidate| {
                if (candidate.inst.ne(ray.exclude0.x) | candidate.prim.ne(ray.exclude0.y))
                    & (candidate.inst.ne(ray.exclude1.x) | candidate.prim.ne(ray.exclude1.y))
                {
                    candidate.commit();
                }
            })
            .trace()
    }
    #[tracked(crate = "luisa")]
    pub fn intersect_hit_info(
        &self,
        ray: Expr<Ray>,
    ) -> (Expr<bool>, Expr<u32>, Expr<u32>, Expr<Float2>) {
        if !self.use_rq {
            let hit = self._trace_closest(ray);
            let inst_id = hit.inst;
            let prim_id = hit.prim;
            let bary = hit.triangle_barycentric_coord();
            (!hit.miss(), inst_id, prim_id, bary)
        } else {
            let hit = self._trace_closest_rq(ray);
            let inst_id = hit.inst;
            let prim_id = hit.prim;
            let bary = hit.triangle_barycentric_coord();
            (hit.triangle_hit(), inst_id, prim_id, bary)
        }
    }
    #[tracked(crate = "luisa")]
    pub fn intersect(&self, ray: Expr<Ray>) -> SurfaceInteraction {
        if !self.use_rq {
            let hit = self._trace_closest(ray);
            if !hit.miss() {
                let inst_id = hit.inst;
                let prim_id = hit.prim;
                let bary = hit.triangle_barycentric_coord();
                self.surface_interaction(inst_id, prim_id, bary)
            } else {
                SurfaceInteraction::invalid()
            }
        } else {
            let hit = self._trace_closest_rq(ray);
            if hit.triangle_hit() {
                let inst_id = hit.inst;
                let prim_id = hit.prim;
                let bary = hit.triangle_barycentric_coord();
                self.surface_interaction(inst_id, prim_id, bary)
            } else {
                SurfaceInteraction::invalid()
            }
        }
    }
    #[tracked(crate = "luisa")]
    pub fn occlude(&self, ray: Expr<Ray>) -> Expr<bool> {
        let ro: Expr<[f32; 3]> = ray.o.into();
        let rd: Expr<[f32; 3]> = ray.d.into();
        let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
        if !self.use_rq {
            self.meshes
                .accel
                .intersect_any(rtx_ray, self.trace_options())
        } else {
            let hit = self
                .meshes
                .accel
                .traverse_any(rtx_ray, self.trace_options())
                .on_surface_hit(|candidate: rtx::SurfaceCandidate| {
                    if (candidate.inst.ne(ray.exclude0.x) | candidate.prim.ne(ray.exclude0.y))
                        & (candidate.inst.ne(ray.exclude1.x) | candidate.prim.ne(ray.exclude1.y))
                    {
                        candidate.commit();
                    }
                })
                .trace();
            !hit.miss()
        }
    }
}
