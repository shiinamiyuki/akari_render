use akari_common::luisa::rtx::SurfaceCandidate;
use luisa::rtx::{AccelTraceOptions, CommittedHit, SurfaceHit};
use std::rc::Rc;
use std::sync::Arc;

use crate::color::ColorPipeline;
use crate::heap::MegaHeap;
use crate::light::LightAggregate;

use crate::svm::surface::Surface;
use crate::svm::Svm;
use crate::util::hash::{xxhash32_2, xxhash32_3, xxhash32_4};
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
    fn alpha_test(&self, candidate: &SurfaceCandidate) -> Expr<bool> {
        let result = false.var();
        outline(|| {
            let si = self.meshes.surface_interaction_for_alpha_test(
                candidate.inst,
                candidate.prim,
                candidate.bary,
            );
            let h = xxhash32_4(Uint4::expr(
                candidate.inst,
                candidate.prim,
                candidate.bary.x.bitcast::<u32>(),
                candidate.bary.y.bitcast::<u32>(),
            ));
            let h = h.as_f32() * (1.0 / u32::MAX as f64) as f32;
            let alpha = self.svm.dispatch_svm(
                si.surface,
                ColorPipeline {
                    color_repr: ColorRepr::Rgb(color::RgbColorSpace::SRgb),
                    rgb_colorspace: color::RgbColorSpace::SRgb,
                },
                si,
                None,
                svm::eval::SvmEvalMode::Alpha,
                |eval| {
                    let shader = eval
                        .eval_shader()
                        .downcast_ref::<Rc<dyn Surface>>()
                        .unwrap()
                        .clone();
                    shader.alpha()
                },
            );
            // device_log!("alpha: {}, hash: {}", alpha, hash);
            *result = (alpha >= 1.0) | (alpha > h);
        });
        **result
    }
    #[tracked(crate = "luisa")]
    pub fn _trace_closest_rq(&self, ray: Expr<Ray>) -> Expr<CommittedHit> {
        // let hit = CommittedHit::var_zeroed();
        // outline(|| {
            let ro: Expr<[f32; 3]> = ray.o.into();
            let rd: Expr<[f32; 3]> = ray.d.into();
            let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
            self
                .meshes
                .accel
                .traverse(rtx_ray, self.trace_options())
                .on_surface_hit(|candidate: rtx::SurfaceCandidate| {
                    if (candidate.inst.ne(ray.exclude0.x) | candidate.prim.ne(ray.exclude0.y))
                        & (candidate.inst.ne(ray.exclude1.x) | candidate.prim.ne(ray.exclude1.y))
                    {
                        if self.alpha_test(&candidate) {
                            candidate.commit();
                        }
                    }
                })
                .trace()
        // });
        // **hit
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
        let result = false.var();
        outline(|| {
            let ro: Expr<[f32; 3]> = ray.o.into();
            let rd: Expr<[f32; 3]> = ray.d.into();
            let rtx_ray = rtx::Ray::new_expr(ro, ray.t_min, rd, ray.t_max);
            *result = if !self.use_rq {
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
                            & (candidate.inst.ne(ray.exclude1.x)
                                | candidate.prim.ne(ray.exclude1.y))
                        {
                            if self.alpha_test(&candidate) {
                                candidate.commit();
                            }
                        }
                    })
                    .trace();
                !hit.miss()
            }
        });
        **result
    }
}
