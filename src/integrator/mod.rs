use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{film::*, scene::*, *};
#[derive(Clone, Debug)]
pub struct RenderOptions {
    pub save_intermediate: bool, // save intermediate results
    pub session: String,
    pub save_stats: bool, // save stats to {session}.json
}
impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            save_intermediate: false,
            session: String::from("default"),
            save_stats: false,
        }
    }
}
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IntermediateStats {
    pub path: String,
    pub time: f64,
    pub spp: u32,
}
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderStats {
    pub intermediate: Vec<IntermediateStats>,
}
pub trait Integrator {
    fn render(&self, scene: Arc<Scene>, film: &mut Film, options: &RenderOptions);
}

pub mod gpt;
pub mod mcmc;
pub mod mcmc_opt;
pub mod normal;
pub mod pt;

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Method {
    #[serde(rename = "pt")]
    PathTracer(pt::Config),
    #[serde(rename = "mcmc")]
    Mcmc(mcmc::Config),
    #[serde(rename = "mcmc_opt")]
    McmcOpt(mcmc::Config),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    pub method: Method,
    pub out: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RenderTask {
    Single(RenderConfig),
    Multi(Vec<RenderConfig>),
}

pub fn render(device: Device, scene: Arc<Scene>, task: &RenderTask, options: RenderOptions) {
    fn render_single(
        device: Device,
        scene: &Arc<Scene>,
        config: &RenderConfig,
        options: &RenderOptions,
    ) {
        let mut film = Film::new(
            device.clone(),
            scene.camera.resolution(),
            FilmColorRepr::SRgb,
        );
        let output_image: luisa::Tex2d<luisa::Float4> = device.create_tex2d(
            luisa::PixelStorage::Float4,
            scene.camera.resolution().x,
            scene.camera.resolution().y,
            1,
        );
        let tic = std::time::Instant::now();
        match &config.method {
            Method::PathTracer(config) => {
                pt::render(device.clone(), scene.clone(), &mut film, &config, &options)
            }
            Method::Mcmc(config) => {
                mcmc::render(device.clone(), scene.clone(), &mut film, &config, &options)
            }
            Method::McmcOpt(config) => {
                mcmc_opt::render(device.clone(), scene.clone(), &mut film, &config, &options)
            }
        }
        let toc = std::time::Instant::now();
        log::info!("Completed in {:.1}ms", (toc - tic).as_secs_f64() * 1e3);
        film.copy_to_rgba_image(&output_image);
        util::write_image(&output_image, &config.out);
    }
    match task {
        RenderTask::Single(config) => render_single(device, &scene, config, &options),
        RenderTask::Multi(configs) => {
            let tic = std::time::Instant::now();
            log::info!("Rendering with {} methods", configs.len());
            for (i, config) in configs.iter().enumerate() {
                log::info!("Rendering task {}/{}", i + 1, configs.len());
                render_single(device.clone(), &scene, config, &options)
            }
            let toc = std::time::Instant::now();
            log::info!("Completed in {:.3}s", (toc - tic).as_secs_f64());
        }
    }
}
