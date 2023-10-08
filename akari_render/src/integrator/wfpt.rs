use std::fmt::Display;

use super::*;
use crate::{
    color::*,
    film::*,
    geometry::*,
    interaction::SurfaceInteraction,
    loop_,
    mesh::MeshInstance,
    sampler::*,
    scene::*,
    svm::surface::{diffuse::DiffuseBsdf, *},
    util::profile::DispatchProfiler,
    *,
};
use pt::PathTracerBase;
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

struct Status;
impl Status {
    const TERMINATED: u32 = 0;
}
#[derive(Clone, Copy, Debug, Soa, Value)]
#[repr(C)]
struct PathState {
    si: SurfaceInteraction,
    prev_ng: Float3,
    depth: u32,
    prev_bsdf_pdf: f32,
    spp: u32,
    status: u32,
}
struct WavefrontPathTracer {
    device: Device,
    radiance: ColorBuffer,
    beta: ColorBuffer,
    direct_li: ColorBuffer,
    states: SoaBuffer<PathState>,
    rays: SoaBuffer<Ray>,
    surface_queue: Buffer<u32>,
    surface_queue_counts: Buffer<u32>,
    scene: Arc<Scene>,
    config: Config,
    config_buffer: Buffer<Config>,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum KernelKind {
    Raygen,
    Intersect,
    SortSurfaceQueue,
    DirectLighting,
    TestShadow,
    ShadeSurface,
}
impl Display for KernelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelKind::Raygen => write!(f, "raygen"),
            KernelKind::Intersect => write!(f, "intersect"),
            KernelKind::SortSurfaceQueue => write!(f, "sort_surface_queue"),
            KernelKind::DirectLighting => write!(f, "direct_lighting"),
            KernelKind::TestShadow => write!(f, "test_shadow"),
            KernelKind::ShadeSurface => write!(f, "shade_surface"),
        }
    }
}
impl WavefrontPathTracer {
    fn new(
        device: Device,
        config: Config,
        color_pipeline: ColorPipeline,
        scene: Arc<Scene>,
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
        let surface_queue = device.create_buffer(config.wavefront_size as usize);
        let surface_queue_offsets =
            device.create_buffer(scene.svm.surface_shaders.variant_count() as usize);
        let rays = device.create_soa_buffer(config.wavefront_size as usize);
        let config_buffer = device.create_buffer_from_slice(&[config.clone()]);
        Self {
            device,
            radiance,
            beta,
            direct_li,
            states,
            surface_queue,
            surface_queue_counts: surface_queue_offsets,
            config_buffer,
            rays,
            scene,
            config,
        }
    }

    /**
     * Generate new camera rays
     * If the current path is terminated, generate new camera rays
     */
    #[tracked]
    fn raygen(&self) {
        let tid = dispatch_id().x;
        let states = self.states.var();
        let spp = states.spp.read(tid);
        let cfg = self.config_buffer.read(0);
        if spp >= cfg.spp {
            return;
        };
        let status = states.status.read(tid);
        // only generate new camera rays if the current path is terminated
        if status != Status::TERMINATED {
            return;
        };
    }

    /**
     * Intersect rays with the scene
     */
    #[tracked]
    fn intersect(&self) {
        let tid = dispatch_id().x;
        let rays = self.rays.var();
        let states = self.states.var();
        let ray = rays.read(tid);
        let si = self.scene.intersect(ray);
        let state = states.read(tid).var();
        *state.si = si;
        states.write(tid, state);
    }
    fn direct_lighting(&self) {}
    fn test_shadow(&self) {}
    fn shade_surface(&self) {}
    fn clear_surface_queue(&self) {
        let tid = dispatch_id().x;
        self.surface_queue_counts.write(tid, 0);
    }
    fn allocate_surface_queue(&self, count_or_allocate: bool) {
        let tid = dispatch_id().x;
    }
    fn sort_surface_queue(&self) {}
    fn render_loop(&self) {
        let raygen_kernel = self.device.create_kernel_async::<fn()>(&|| {
            self.raygen();
        });
        let intersect_kernel = self.device.create_kernel_async::<fn()>(&|| {
            self.intersect();
        });
        let direct_lighting_kernel = self.device.create_kernel_async::<fn()>(&|| {
            self.direct_lighting();
        });
        let test_shadow_kernel = self.device.create_kernel_async::<fn()>(&|| {
            self.test_shadow();
        });
        let shade_surface_kernel = self.device.create_kernel_async::<fn()>(&|| {
            self.shade_surface();
        });
        let clear_surface_queue_kernel = self.device.create_kernel_async::<fn()>(&|| {
            self.clear_surface_queue();
        });
        let allocate_surface_queue_kernel = self.device.create_kernel_async::<fn()>(&|| {
            // self.allocate_surface_queue();
        });
        let sort_surface_queue_kernel = self.device.create_kernel_async::<fn()>(&|| {
            self.sort_surface_queue();
        });

        let npixels =
            (self.scene.camera.resolution().x * self.scene.camera.resolution().y) as usize;
        let total_rays = npixels * self.config.spp as usize;
        let nlaunches = (total_rays + self.config.wavefront_size as usize - 1)
            / self.config.wavefront_size as usize;
        let nloops = nlaunches * (self.config.max_depth as usize + 1);
        let profiler = DispatchProfiler::<KernelKind>::new();
        let wf_sz = self.config.wavefront_size as u32;
        self.device.default_stream().with_scope(|s| {
            for spp in 0..nloops {
                for _ in 0..(self.config.max_depth + 1) {
                    profiler.profile(KernelKind::Raygen, s, || {
                        s.submit([raygen_kernel.dispatch_async([wf_sz, 1, 1])]);
                    });
                    profiler.profile(KernelKind::Intersect, s, || {
                        s.submit([intersect_kernel.dispatch_async([wf_sz, 1, 1])]);
                    });
                    profiler.profile(KernelKind::SortSurfaceQueue, s, || {
                        s.submit([clear_surface_queue_kernel.dispatch_async([
                            self.surface_queue_counts.len() as u32,
                            1,
                            1,
                        ])]);
                    });
                    profiler.profile(KernelKind::DirectLighting, s, || {
                        s.submit([direct_lighting_kernel.dispatch_async([wf_sz, 1, 1])]);
                    });
                    profiler.profile(KernelKind::TestShadow, s, || {
                        s.submit([test_shadow_kernel.dispatch_async([wf_sz, 1, 1])]);
                    });
                    profiler.profile(KernelKind::ShadeSurface, s, || {
                        s.submit([shade_surface_kernel.dispatch_async([wf_sz, 1, 1])]);
                    });
                }
                if (spp + 1) % self.config.spp_per_pass as usize == 0 {
                    s.synchronize();
                }
            }
        });
    }
}
