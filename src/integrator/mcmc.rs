use rand::{thread_rng, Rng};

use super::pt::PathTracer;
use super::Integrator;
use crate::{
    color::*, film::*, geometry::*, interaction::*, sampler::*, scene::*, surface::Bsdf, *,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Simple,
    LangevinOnline,
    LangevinHybrid,
}
pub struct MCMC {
    pub device: Device,
    pub spp: u32,
    pub max_depth: u32,
    pub method: Method,
    pub n_chains: usize,
    pub n_bootstrap: usize,
}

impl MCMC {
    pub fn new(
        device: Device,
        spp: u32,
        max_depth: u32,
        method: Method,
        n_chains: usize,
        n_bootstrap: usize,
    ) -> Self {
        Self {
            device,
            spp,
            max_depth,
            method,
            n_chains,
            n_bootstrap,
        }
    }
    fn bootstrap(&self, scene: &Scene, seeds: &Buffer<u32>) -> luisa::Result<Buffer<u32>> {
        let mut rng = thread_rng();
        let seeds = self
            .device
            .create_buffer_from_fn(self.n_bootstrap, |_| rng.gen::<u32>())?;
        self.device
            .create_kernel::<()>(&|| {
                let i = dispatch_id().x();
                let seed = seeds.var().read(i);
                let lcg_sampler = LcgSampler {
                    state: var!(u32, seed),
                };
            })?
            .dispatch([self.n_bootstrap as u32, 1, 1])?;
        todo!()
    }
}
impl Integrator for MCMC {
    fn render(&self, scene: &Scene, film: &mut Film) -> luisa::Result<()> {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}, spp: {}",
            resolution.x,
            resolution.y,
            self.spp
        );

        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        film.clear();
        Ok(())
    }
}
