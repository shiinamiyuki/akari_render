use std::{sync::Arc, time::Instant};

use super::{Integrator, RenderSession};
use crate::{color::*, film::*, sampler::*, scene::*, *};
use akari_render::svm::surface::{BsdfEvalContext, Surface};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
pub enum Aov {
    #[serde(rename = "ns")]
    ShadingNormal,
    #[serde(rename = "ng")]
    GeometryNormal,
    #[serde(rename = "tangent")]
    Tangent,
    #[serde(rename = "bitangent")]
    Bitangent,
    #[serde(rename = "albedo")]
    Albedo,
    #[serde(rename = "roughness")]
    Roughness,
}
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct Config {
    pub spp: u32,
    pub aov: Aov,
    pub remap: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            spp: 256,
            aov: Aov::ShadingNormal,
            remap: true,
        }
    }
}
pub struct NormalVis {
    device: Device,
    spp: u32,
    aov: Aov,
    remap: bool,
}
impl NormalVis {
    pub fn new(device: Device, spp: u32, aov: Aov, remap: bool) -> Self {
        Self {
            device,
            spp,
            aov,
            remap,
        }
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
                        // let ns = si.frame.t;
                        // Color::Rgb(ns * 0.5 + 0.5, color_repr.rgb_colorspace().unwrap())
                        match self.aov {
                            Aov::ShadingNormal => {
                                let v = scene.svm.dispatch_surface(
                                    si.surface,
                                    color_pipeline,
                                    si,
                                    swl,
                                    |closure| closure.ns(),
                                );
                                let v = if self.remap { v * 0.5 + 0.5 } else { v };
                                Color::Rgb(v, color_repr.rgb_colorspace().unwrap())
                            }
                            Aov::GeometryNormal => {
                                let v = si.ng;
                                let v = if self.remap { v * 0.5 + 0.5 } else { v };
                                Color::Rgb(v, color_repr.rgb_colorspace().unwrap())
                            }
                            Aov::Tangent => {
                                let v = si.frame.t;
                                let v = if self.remap { v * 0.5 + 0.5 } else { v };
                                Color::Rgb(v, color_repr.rgb_colorspace().unwrap())
                            }
                            Aov::Bitangent => {
                                let v = si.frame.s;
                                let v = if self.remap { v * 0.5 + 0.5 } else { v };
                                Color::Rgb(v, color_repr.rgb_colorspace().unwrap())
                            }
                            Aov::Albedo => scene.svm.dispatch_surface(
                                si.surface,
                                color_pipeline,
                                si,
                                swl,
                                |closure| {
                                    let ctx = BsdfEvalContext {
                                        color_repr,
                                        ad_mode: ADMode::None,
                                    };
                                    closure.albedo(-ray.d, swl, &ctx)
                                        + closure.emission(-ray.d, swl, &ctx)
                                },
                            ),

                            Aov::Roughness => scene.svm.dispatch_surface(
                                si.surface,
                                color_pipeline,
                                si,
                                swl,
                                |closure| {
                                    let ctx = BsdfEvalContext {
                                        color_repr,
                                        ad_mode: ADMode::None,
                                    };
                                    Color::one(color_repr)
                                        * closure.roughness(-ray.d, sampler.next_1d(), swl, &ctx)
                                },
                            ),
                        }
                    } else {
                        Color::zero(color_repr)
                    };
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
    let pt = NormalVis::new(device.clone(), config.spp, config.aov, config.remap);
    pt.render(scene, sampler, color_pipeline, film, options);
}
