use std::{sync::Arc, time::Instant};

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
        _options: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}, spp: {}",
            resolution.x,
            resolution.y,
            self.spp
        );
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        let sampler_creator = sampler.creator(self.device.clone(), &scene, self.spp);
        let kernel = self
            .device
            .create_kernel::<fn(u32)>(track!(&|spp: Expr<u32>| {
                let p = dispatch_id().xy();
                let sampler = sampler_creator.create(p);
                let sampler = sampler.as_ref();
                let color_repr = color_pipeline.color_repr;
                for _ in 0u32.expr()..spp {
                    let swl = sample_wavelengths(color_repr, sampler);
                    sampler.start();
                    let (ray, ray_w) = scene.camera.generate_ray(
                        &scene,
                        film.filter(),
                        p,
                        sampler,
                        color_repr,
                        swl,
                    );
                    let si = scene.intersect(ray);

                    let color = if si.valid {
                        let ns = si.ns();
                        Color::Rgb(ns * 0.5 + 0.5, color_repr.rgb_colorspace().unwrap())
                    } else {
                        Color::zero(color_repr)
                    };
                    // let color = if !hit.miss() {
                    //     let rgb = Float3::expr(1.0, hit.u, hit.v);
                    //     Color::Rgb(rgb, color_repr.rgb_colorspace().unwrap()) * ray_color
                    // } else {
                    //     Color::zero(color_repr)
                    // };
                    // let color = {
                    //     let rgb = ray.d * 0.5 + 0.5;
                    //     Color::Rgb(rgb, color_repr.rgb_colorspace().unwrap()) * ray_color
                    // };
                    film.add_sample(p.cast_f32(), &color, swl, ray_w);
                }
            }));
        let stream = self.device.create_stream(StreamTag::Graphics);
        let clk = Instant::now();
        stream.with_scope(|s| {
            let mut cmds = vec![];
            cmds.push(kernel.dispatch_async([resolution.x, resolution.y, 1], &self.spp));
            s.submit(cmds);
            s.synchronize();
        });
        let elapsed = clk.elapsed().as_secs_f64();
        log::info!("Rendered in {:.2}ms", elapsed * 1000.0);
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
