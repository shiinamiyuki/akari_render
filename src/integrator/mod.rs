use serde::{Deserialize, Serialize};

use crate::{film::*, scene::*, *};

pub trait Integrator {
    fn render(&self, scene: &Scene, film: &mut Film);
}

pub mod gpt;
pub mod mcmc;
pub mod normal;
pub mod pt;

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Method {
    #[serde(rename = "pt")]
    PathTracer(pt::Config),
    #[serde(rename = "mcmc")]
    Mcmc(mcmc::Config),
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

pub fn render(device: Device, scene: &Scene, task: &RenderTask) {
    fn render_single(device: Device, scene: &Scene, config: &RenderConfig) {
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
            Method::PathTracer(config) => pt::render(device.clone(), scene, &mut film, &config),
            Method::Mcmc(config) => mcmc::render(device.clone(), scene, &mut film, &config),
        }
        let toc = std::time::Instant::now();
        log::info!("Rendered in {:.1}ms", (toc - tic).as_secs_f64() * 1e3);
        film.copy_to_rgba_image(&output_image);
        util::write_image(&output_image, &config.out);
    }
    match task {
        RenderTask::Single(config) => render_single(device, scene, config),
        RenderTask::Multi(configs) => {
            let tic = std::time::Instant::now();
            log::info!("Rendering with {} methods", configs.len());
            for (i, config) in configs.iter().enumerate() {
                log::info!("Rendering {}/{}", i + 1, configs.len());
                render_single(device.clone(), scene, config)
            }
            let toc = std::time::Instant::now();
            log::info!("Rendered in {:.3}s", (toc - tic).as_secs_f64());
        }
    }
}
