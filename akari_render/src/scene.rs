use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use luisa::resource::IoTexel;
use luisa::rtx::{CommittedHit, Hit};

use crate::color::{Color, ColorPipeline, ColorRepr, FlatColor, SampledWavelengths};

use crate::light::{LightAggregate, LightEvalContext};

use crate::svm::{ShaderRef, Svm};
use crate::{camera::Camera, geometry::*, interaction::*, mesh::*, *};

pub struct ResourceHeap<K: Hash + Eq> {
    device: Device,
    bindless: BindlessArray,
    tex2ds: HashMap<K, u32>,
    tex3ds: HashMap<K, u32>,
    buffers: HashMap<K, u32>,
}

impl<K: Hash + Eq> ResourceHeap<K> {
    pub fn new(device: Device, count: usize) -> Self {
        let bindless = device.create_bindless_array(count);
        Self {
            device,
            bindless,
            tex2ds: HashMap::new(),
            tex3ds: HashMap::new(),
            buffers: HashMap::new(),
        }
    }
    pub fn push_buffer<T: Value>(&mut self, key: K, buffer: &Buffer<T>) {
        assert!(!self.buffers.contains_key(&key));
        let index = self.buffers.len();
        self.bindless.emplace_buffer_async(index, buffer);
    }
    pub fn get_buffer_index(&self, key: &K) -> Option<u32> {
        self.buffers.get(key).copied()
    }
    pub fn push_tex2d<T: IoTexel>(&mut self, key: K, tex: &Tex2d<T>, sampler: TextureSampler) {
        assert!(!self.tex2ds.contains_key(&key));
        let index = self.tex2ds.len();
        self.bindless.emplace_tex2d_async(index, tex, sampler);
    }
    pub fn get_tex2d_index(&self, key: &K) -> Option<u32> {
        self.tex2ds.get(key).copied()
    }
    pub fn push_tex3d<T: IoTexel>(&mut self, key: K, tex: &Tex3d<T>, sampler: TextureSampler) {
        assert!(!self.tex3ds.contains_key(&key));
        let index = self.tex3ds.len();
        self.bindless.emplace_tex3d_async(index, tex, sampler);
    }
    pub fn commit(&mut self) {
        self.bindless.update();
    }
    pub fn bindless(&self) -> &BindlessArray {
        &self.bindless
    }
    pub fn reset(&mut self) {
        for i in 0..self.tex2ds.len() {
            self.bindless.remove_tex2d_async(i);
        }
        for i in 0..self.tex3ds.len() {
            self.bindless.remove_tex3d_async(i);
        }
        for i in 0..self.buffers.len() {
            self.bindless.remove_buffer_async(i);
        }
        if !self.tex2ds.is_empty() || !self.tex3ds.is_empty() || !self.buffers.is_empty() {
            self.bindless.update();
        }
        self.tex2ds.clear();
        self.tex3ds.clear();
        self.buffers.clear();
    }
}

pub struct Scene {
    pub svm: Arc<Svm>,
    pub lights: LightAggregate,
    pub meshes: Arc<MeshAggregate>,
    pub camera: Arc<dyn Camera>,
    pub device: Device,
    pub use_rq: bool,
    pub printer: Printer,
    // pub heap: ResourceHeap<String>,
    // pub tmp_heap: ResourceHeap<String>,
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
    pub fn intersect_hit_info(
        &self,
        ray: Expr<Ray>,
    ) -> (Expr<bool>, Expr<u32>, Expr<u32>, Expr<Float2>) {
        if !self.use_rq {
            let hit = self._trace_closest(ray);
            let inst_id = hit.inst_id;
            let prim_id = hit.prim_id;
            let bary = Float2::expr(hit.u, hit.v);
            (!hit.miss(), inst_id, prim_id, bary)
        } else {
            let hit = self._trace_closest_rq(ray);
            let inst_id = hit.inst_id;
            let prim_id = hit.prim_id;
            let bary = hit.bary;
            (hit.triangle_hit(), inst_id, prim_id, bary)
        }
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
