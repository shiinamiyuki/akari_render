use std::fmt::Display;

use super::{pt::PathTracerBase, *};
use crate::{
    color::*,
    geometry::*,
    interaction::SurfaceInteraction,
    sampler::{Sampler, SamplerCreator},
    svm::ShaderRef,
    util::profile::DispatchProfiler,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize, Debug, Value)]
#[serde(default)]
#[repr(C)]
pub struct Config {
    pub spp: u32,
    pub max_depth: u32,
    pub spp_per_pass: u32,
    pub use_nee: bool,
    pub rr_depth: u32,
    pub indirect_only: bool,
    pub force_diffuse: bool,
    pub wavefront_size: u32,
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
            wavefront_size: 512 * 512,
        }
    }
}

#[derive(Clone, Copy, Debug, Soa, Value)]
#[repr(C)]
struct SurfaceInteractionData {
    pub frame: Frame,
    pub p: Float3,
    pub ng: Float3,
    pub bary: Float2,
    pub uv: Float2,
    pub inst_id: u32,
    pub prim_id: u32,
    pub surface: ShaderRef,
    pub prim_area: f32,
}
#[derive(Clone, Copy, Debug, Soa, Value)]
#[repr(C)]
struct PathState {
    sid: u32,
    swl: SampledWavelengths,
    si: SurfaceInteractionData,
    pixel: Uint2,
    depth: u32,
    prev_bsdf_pdf: f32,
    prev_ng: Float3,
    spp: u32,
    ray_weight: f32,
    ray: Ray,
    shadow_ray: Ray,
}
struct WavePathImpl {
    device: Device,
    /// only write atomically into this buffer
    radiance: ColorBuffer,
    beta: ColorBuffer,
    direct_li: ColorBuffer,
    states: SoaBuffer<PathState>,
    work_queue: KernelWorkQueue,
    surface_queue: KernelWorkQueue,
    scene: Arc<Scene>,
    config: Config,
    config_buffer: Buffer<Config>,
    color_pipeline: ColorPipeline,
    sampler_creator: Box<dyn SamplerCreator>,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(u32)]
enum KernelId {
    GenerateRay = 0,
    Intersect,
    Miss,
    SortSurface,
    ShadeSurface,
    /// test shadow can be enqueued in parallel with other kernels
    /// except ShadeSurface, which can enqueue new TestShadow
    TestShadow,

    Total,
}
impl Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelId::GenerateRay => write!(f, "raygen"),
            KernelId::Intersect => write!(f, "intersect"),
            KernelId::Miss => write!(f, "miss"),
            KernelId::TestShadow => write!(f, "test_shadow"),
            KernelId::SortSurface => write!(f, "sort_surface"),
            KernelId::ShadeSurface => write!(f, "shade_surface"),
            KernelId::Total => unreachable!(),
        }
    }
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum KernelWorkQueueSortMode {
    AtomicAdd,
    CountAndAllocate,
}
/// Queue for accumulating kernel work items
struct KernelWorkQueue {
    max_size: usize,
    counter: Buffer<u32>,
    offsets: Buffer<u32>,
    /// if sort_mode == AtomicAdd:
    /// [u32 x (max_size x #variants)]
    /// if sort_mode == CountAndAllocate:
    /// [u32 x max_size]
    queues: Buffer<u32>,
    reset_kernel: Kernel<fn(u32)>,
    alloc_kernel: Kernel<fn()>,
    num_variants: u32,
    sort_mode: KernelWorkQueueSortMode,
}
impl KernelWorkQueue {
    fn new(
        device: &Device,
        num_variants: u32,
        max_size: usize,
        sort_mode: KernelWorkQueueSortMode,
    ) -> Self {
        let counter = device.create_buffer(num_variants as usize);
        let offsets = device.create_buffer(num_variants as usize);
        let queues = device.create_buffer(if sort_mode == KernelWorkQueueSortMode::AtomicAdd {
            max_size * num_variants as usize
        } else {
            max_size
        });
        let reset_kernel = device.create_kernel_async::<fn(u32)>(&track!(|id| {
            if id == u32::MAX {
                for i in 0..num_variants {
                    counter.write(i, 0);
                }
            } else {
                counter.write(id, 0);
            }
        }));
        let alloc_kernel = device.create_kernel_async::<fn()>(&track!(|| {
            let sum = 0u32.var();
            for i in 0..num_variants {
                let c = counter.read(i);
                offsets.write(i, sum);
                *sum += c;
            }
        }));
        Self {
            max_size,
            counter,
            offsets,
            queues,
            reset_kernel,
            alloc_kernel,
            num_variants,
            sort_mode,
        }
    }
    #[tracked]
    fn enqueue(&self, kid: u32, sid: Expr<u32>) {
        assert!(kid < self.num_variants);
        self.enqueue_dynamic(kid.expr(), sid)
    }

    #[tracked]
    fn enqueue_dynamic(&self, kid: Expr<u32>, sid: Expr<u32>) {
        if debug_mode() {
            lc_assert!(kid.lt(self.num_variants as u32));
        }
        let counter = self.counter.var();
        let i = counter.atomic_fetch_and(kid, 1);
        let queues = self.queues.var();
        if self.sort_mode == KernelWorkQueueSortMode::AtomicAdd {
            queues.write(i + (kid * self.max_size as u32), sid);
        }
    }

    #[tracked]
    fn sort_dynamic(&self, kid: Expr<u32>, sid: Expr<u32>) {
        assert_eq!(self.sort_mode, KernelWorkQueueSortMode::CountAndAllocate);
        if debug_mode() {
            lc_assert!(kid.lt(self.num_variants as u32));
        }
        let offset = self.counter.var().atomic_fetch_and(kid, 1);
        let queues = self.queues.var();
        queues.write(offset, sid);
    }
    fn reset(&self, kid: u32, scope: &Scope) {
        scope.submit([self.reset_kernel.dispatch_async([1, 1, 1], &(kid as u32))]);
    }
    fn compute_offsets(&self, kid: u32, scope: &Scope) {
        assert_eq!(self.sort_mode, KernelWorkQueueSortMode::CountAndAllocate);
        scope.submit([self.alloc_kernel.dispatch_async([1, 1, 1])]);
    }
}

impl WavePathImpl {
    #[tracked]
    fn init_pt_impl(
        &self,
        swl: Var<SampledWavelengths>,
        depth: Expr<u32>,
        prev_bsdf_pdf: Expr<f32>,
        prev_ng: Expr<Float3>,
    ) -> PathTracerBase {
        let pt = PathTracerBase::new(
            &self.scene,
            self.color_pipeline,
            self.config.max_depth.expr(),
            self.config.rr_depth.expr(),
            self.config.use_nee,
            self.config.indirect_only,
            swl,
        );
        *pt.depth = depth;
        *pt.prev_bsdf_pdf = prev_bsdf_pdf;
        *pt.prev_ng = prev_ng;
        pt
    }
    fn new(
        device: Device,
        config: Config,
        color_pipeline: ColorPipeline,
        scene: Arc<Scene>,
        sampler: SamplerConfig,
    ) -> Self {
        let radiance = ColorBuffer::new(
            device.clone(),
            config.wavefront_size as usize,
            color_pipeline.color_repr,
        );
        let beta = ColorBuffer::new(
            device.clone(),
            config.wavefront_size as usize,
            color_pipeline.color_repr,
        );
        let direct_li = ColorBuffer::new(
            device.clone(),
            config.wavefront_size as usize,
            color_pipeline.color_repr,
        );
        let states = device.create_soa_buffer(config.wavefront_size as usize);
        let surface_queue = KernelWorkQueue::new(
            &device,
            scene.svm.surface_shaders.variant_count() as u32,
            config.wavefront_size as usize,
            KernelWorkQueueSortMode::CountAndAllocate,
        );
        let config_buffer = device.create_buffer_from_slice(&[config.clone()]);
        let work_queue = KernelWorkQueue::new(
            &device,
            KernelId::Total as u32,
            config.wavefront_size as usize,
            KernelWorkQueueSortMode::AtomicAdd,
        );
        let sampler_creator = sampler.creator(device.clone(), &scene, config.spp);
        Self {
            device,
            radiance,
            beta,
            direct_li,
            states,
            surface_queue,
            config_buffer,
            scene,
            work_queue,
            config,
            sampler_creator,
            color_pipeline,
        }
    }

    /**
     * Generate new camera rays
     * If the current path is terminated, generate new camera rays
     */
    #[tracked]
    fn raygen(&self, film: &Film) {
        let tid = dispatch_id().x;
        let states = self.states.var();
        let sid = states.sid.read(tid);
        if debug_mode() {
            let spp = states.spp.read(sid);
            let cfg = self.config_buffer.read(0);
            lc_assert!(spp.lt(cfg.spp));
        }

        let pixel = states.pixel.read(sid);
        let sampler = self.sampler_creator.create(pixel);
        let sampler = sampler.as_ref();
        sampler.start();
        let swl = sample_wavelengths(self.color_pipeline.color_repr, sampler);
        let (ray, w) = self.scene.camera.generate_ray(
            &self.scene,
            film.filter(),
            pixel,
            sampler,
            self.color_pipeline.color_repr,
            swl,
        );
        states.ray.write(sid, ray);
        states.ray_weight.write(sid, w);
        self.work_queue.enqueue(KernelId::Intersect as u32, sid);
    }

    /**
     * Intersect rays with the scene
     */
    #[tracked]
    fn intersect(&self) {
        let tid = dispatch_id().x;
        let states = self.states.var();
        let sid = states.sid.read(tid);
        let ray = states.ray.read(sid);
        let si = self.scene.intersect(ray);

        if si.valid {
            states.si.bary.write(sid, si.bary);
            states.si.frame.write(sid, si.frame);
            states.si.inst_id.write(sid, si.inst_id);
            states.si.prim_id.write(sid, si.prim_id);
            states.si.prim_area.write(sid, si.prim_area);
            states.si.uv.write(sid, si.uv);
            states.si.ng.write(sid, si.ng);
            states.si.p.write(sid, si.p);
            self.work_queue.enqueue(KernelId::SortSurface as u32, sid);
        } else {
            self.work_queue.enqueue(KernelId::Miss as u32, sid);
        }
    }
    fn load_swl(&self, sid: Expr<u32>) -> Var<SampledWavelengths> {
        if self.color_pipeline.color_repr != ColorRepr::Spectral {
            SampledWavelengthsExpr::rgb_wavelengths().var()
        } else {
            self.states.var().swl.read(sid).var()
        }
    }
    fn load_si(&self, sid: Expr<u32>) -> SurfaceInteraction {
        let states = self.states.var();
        let si = states.si.read(sid);
        SurfaceInteraction {
            bary: si.bary,
            frame: si.frame,
            inst_id: si.inst_id,
            ng: si.ng,
            p: si.p,
            prim_area: si.prim_area,
            prim_id: si.prim_id,
            uv: si.uv,
            surface: si.surface,
            valid: true.expr(),
        }
    }
    fn load_sampler(&self, sid: Expr<u32>) -> Box<dyn Sampler> {
        let states = self.states.var();
        let sampler = self.sampler_creator.create(states.pixel.read(sid));
        sampler
    }
    #[tracked]
    fn shade_surface(&self, shader_kind: Option<u32>) {
        let tid = dispatch_id().x;
        let states = self.states.var();
        let sid = states.sid.read(tid);
        let swl = self.load_swl(sid);
        let prev_bsdf_pdf = states.prev_bsdf_pdf.read(sid);
        let prev_ng = states.prev_ng.read(sid);
        let depth = states.depth.read(sid);
        let pt = self.init_pt_impl(swl, depth, prev_bsdf_pdf, prev_ng);
        let si = self.load_si(sid);
        let sampler = self.load_sampler(sid);
        let sampler = sampler.as_ref();
        let dl = pt.sample_light(si, sampler.next_3d());
        if dl.valid {
            self.work_queue.enqueue(KernelId::TestShadow as u32, sid);
        }
        let wo = -states.ray.read(sid).d;
        let (bsdf_sample, direct) =
            pt.sample_surface_and_shade_direct(shader_kind, si, wo, dl, sampler.next_3d());

        let (beta, _) = self.beta.read(sid);
        self.direct_li.write(sid, direct * beta, **swl);
        states.swl.write(sid, swl);
        if bsdf_sample.valid {
            self.work_queue.enqueue(KernelId::Intersect as u32, sid);
        } else {
            // terminated
            self.work_queue.enqueue(KernelId::GenerateRay as u32, sid);
        }
    }

    #[tracked]
    fn test_shadow(&self) {
        let tid = dispatch_id().x;
        let states = self.states.var();
        let sid = states.sid.read(tid);
        let shadow_ray = states.shadow_ray.read(sid);
        let occluded = self.scene.occlude(shadow_ray);
        if !occluded {
            let (direct, _) = self.direct_li.read(sid);
            self.radiance.atomic_add(sid, direct);
        }
    }
    #[tracked]
    fn sort_surcace(&self, counting: bool) {
        let tid = dispatch_id().x;
        let states = self.states.var();
        let sid = states.sid.read(tid);
        let surface = states.si.surface.read(sid);
        if counting {
            self.surface_queue.enqueue_dynamic(surface.shader_kind, sid);
        } else {
            self.surface_queue.sort_dynamic(surface.shader_kind, sid);
        }
    }
    fn render_loop(&self, film: &Film) {
        let npixels =
            (self.scene.camera.resolution().x * self.scene.camera.resolution().y) as usize;
        let total_rays = npixels * self.config.spp as usize;

        let mut max_wf_sz = self.config.wavefront_size as u32;
        if max_wf_sz >= npixels as u32 {
            log::warn!(
                "Wavefront size {} is larger than the number of pixels {}. Reducing to {}.",
                max_wf_sz,
                npixels,
                npixels
            );
            max_wf_sz = npixels as u32;
        }
        let stream = self.device.create_stream(StreamTag::Graphics);
        let schedule_kernels = |pixel_offset: u32, wf_sz: u32| {
            assert!(wf_sz < max_wf_sz);
            assert!(pixel_offset + wf_sz <= npixels as u32);
        };
        
        let profiler = DispatchProfiler::<KernelId>::new();
    }
}

pub struct WavefrontPathTracer {
    device: Device,
    config: Config,
}
impl WavefrontPathTracer {
    pub fn new(device: Device, config: Config) -> Self {
        Self { device, config }
    }
}

impl Integrator for WavefrontPathTracer {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        options: &RenderSession,
    ) {
        let imp = WavePathImpl::new(
            self.device.clone(),
            self.config,
            color_pipeline,
            scene,
            sampler,
        );
        imp.render_loop(film);
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
    let wfpt = WavefrontPathTracer::new(device.clone(), config.clone());
    wfpt.render(scene, sampler, color_pipeline, film, options);
}
