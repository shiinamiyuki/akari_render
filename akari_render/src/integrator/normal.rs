use std::sync::Arc;

use super::{Integrator, RenderSession};
use crate::{color::*, film::*, sampler::*, scene::*, *};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Config {
    pub spp: u32,
}
impl Default for Config {
    fn default() -> Self {
        Self { spp: 256 }
    }
}
pub struct NormalVis {
    device: Device,
    pub spp: u32,
}
impl NormalVis {
    pub fn new(device: Device, spp: u32) -> Self {
        Self { device, spp }
    }
}

impl Integrator for NormalVis {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        options: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}, spp: {}",
            resolution.x,
            resolution.y,
            self.spp
        );
        let npixels = resolution.x as usize * resolution.y as usize;
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        let sampler_creator =
            IndependentSamplerCreator::new(self.device.clone(), scene.camera.resolution(), 0);
        let kernel = self
            .device
            .create_kernel::<fn(u32)>(track!(&|_spp: Expr<u32>| {
                let p = dispatch_id().xy();
                let sampler = sampler_creator.create(p);
                let sampler = sampler.as_ref();
                let color_repr = color_pipeline.color_repr;
                let swl = sample_wavelengths(color_repr, sampler);
                sampler.start();
                let (ray, ray_color, ray_w) =
                    scene
                        .camera
                        .generate_ray(film.filter(), p, sampler, color_repr, swl);
                let si = scene.intersect(ray);
                // cpu_dbg!(ray);
                let color = if si.valid {
                    let ns = si.ns();
                    // cpu_dbg!(Uint2::expr(si.inst_id(), si.prim_id()));
                    Color::Rgb(ns * 0.5 + 0.5, color_repr.rgb_colorspace().unwrap()) * ray_color
                    // Color::Rgb(Float3::expr(si.bary().x,si.bary().y, 1.0))
                } else {
                    Color::zero(color_repr)
                };
                film.add_sample(p.cast_f32(), &color, swl, ray_w);
            }));
        let stream = self.device.default_stream();
        stream.with_scope(|s| {
            let mut cmds = vec![];
            for _ in 0..self.spp {
                cmds.push(kernel.dispatch_async([resolution.x, resolution.y, 1], &self.spp));
            }
            s.submit(cmds);
            s.synchronize();
        });
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
    let pt = NormalVis::new(device.clone(), config.spp);
    pt.render(scene, sampler, color_pipeline, film, options);
}
