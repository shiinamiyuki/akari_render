use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    color::{Color, ColorPipeline, ColorRepr, RgbColorSpace},
    film::*,
    gui::DisplayChannel,
    sampler::SamplerConfig,
    scene::*,
    *,
};

#[derive(Clone)]
pub struct RenderSession {
    pub save_intermediate: bool, // save intermediate results
    pub name: String,
    pub save_stats: bool, // save stats to {session}.json
    pub display: Option<DisplayChannel>,
}
impl Default for RenderSession {
    fn default() -> Self {
        Self {
            save_intermediate: false,
            name: String::from("default"),
            save_stats: false,
            display: None,
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
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        options: &RenderSession,
    );
}

// pub mod gpt;
pub mod mcmc;
// pub mod mcmc_opt;
pub mod wfpt;
// pub mod mcmc_s;
pub mod normal;
pub mod pt;

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Method {
    #[serde(rename = "normal")]
    NormalVis(normal::Config),
    #[serde(rename = "pt")]
    PathTracer(pt::Config),
    // #[serde(rename = "gpt")]
    // GradientPathTracer(gpt::Config),
    // #[serde(rename = "mcmc")]
    // Mcmc(mcmc::Config),
    // #[serde(rename = "mcmc_s")]
    // McmcSinglePath(mcmc_s::Config),
    // #[serde(rename = "mcmc_opt")]
    // McmcOpt(mcmc::Config),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FilmConfig {
    pub out: String,
    pub filter: PixelFilter,
    pub color: FilmColorRepr,
}
impl Default for FilmConfig {
    fn default() -> Self {
        Self {
            out: String::from("out.exr"),
            filter: PixelFilter::default(),
            color: FilmColorRepr::SRgb,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    pub method: Method,
    #[serde(default)]
    pub color: ColorPipeline,
    #[serde(default = "SamplerConfig::default")]
    pub sampler: SamplerConfig,
    pub film: FilmConfig,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RenderTask {
    Single(RenderConfig),
    Multi(Vec<RenderConfig>),
}

pub fn render(device: Device, scene: Arc<Scene>, task: &RenderTask, options: RenderSession) {
    fn render_single(
        device: Device,
        scene: &Arc<Scene>,
        config: &RenderConfig,
        options: &RenderSession,
    ) {
        let mut film = Film::new(
            device.clone(),
            scene.camera.resolution(),
            FilmColorRepr::SRgb,
            config.film.filter,
        );
        let output_image: Tex2d<Float4> = device.create_tex2d(
            PixelStorage::Float4,
            scene.camera.resolution().x,
            scene.camera.resolution().y,
            1,
        );
        log::info!("Rendering to {:?}", config.film);
        let tic = std::time::Instant::now();
        match &config.method {
            Method::PathTracer(c) => pt::render(
                device.clone(),
                scene.clone(),
                config.sampler,
                config.color,
                &mut film,
                &c,
                &options,
            ),
            // Method::GradientPathTracer(c) => gpt::render(
            //     device.clone(),
            //     scene.clone(),
            //     config.sampler,
            //     color_repr,
            //     &mut film,
            //     &c,
            //     &options,
            // ),
            // Method::Mcmc(c) => mcmc::render(
            //     device.clone(),
            //     scene.clone(),
            //     config.sampler,
            //     config.color,
            //     &mut film,
            //     &c,
            //     &options,
            // ),
            // Method::McmcSinglePath(c) => mcmc_s::render(
            //     device.clone(),
            //     scene.clone(),
            //     config.sampler,
            //     config.color,
            //     &mut film,
            //     &c,
            //     &options,
            // ),
            // Method::McmcOpt(c) => mcmc_opt::render(
            //     device.clone(),
            //     scene.clone(),
            //     config.sampler,
            //     config.color,
            //     &mut film,
            //     &c,
            //     &options,
            // ),
            Method::NormalVis(c) => normal::render(
                device.clone(),
                scene.clone(),
                config.sampler,
                config.color,
                &mut film,
                &c,
                &options,
            ),
        }
        let toc = std::time::Instant::now();
        log::info!("Completed in {:.1}ms", (toc - tic).as_secs_f64() * 1e3);
        film.copy_to_rgba_image(&output_image, true);
        util::write_image(&output_image, &config.film.out);
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
