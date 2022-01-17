use crate::*;
use sampler::{Sampler, SobolSampler};

// Streaming Path Tracer

pub struct SteamPathTracer {
    pub spp: u32,
    pub max_depth: u32,
    pub batch_size: usize,
    pub sort_rays: bool,
}

#[derive(Clone, Copy)]
struct PathState {
    sampler: SobolSampler,
    ray: Ray,
    l: Spectrum,
    beta: Spectrum,
    prev_n:Vec3,
    prev_bsdf_pdf:f32,
    is_delta:bool,
}

impl StreamPathTracer {
    pub fn generate_rays(&self) -> Vec<PathState> {}
}
