use akari::api::OocOptions;
// use akari::accel::*;
// use akari::bsdf::*;
// use akari::camera::*;
// use akari::integrator::*;
// use akari::light::*;
use akari::util::LocalFileResolver;
use akari::*;
#[cfg(feature = "gpu")]
use vkc::Context;
// use akari::shape::*;
// use akari::*;
use std::path::Path;
use std::process::exit;
// use std::sync::Arc;
extern crate clap;
use akari::film::Film;
use akari::profile_fn;
use akari::{api, rayon};
use clap::{App, Arg};

use log::{Level, Metadata, Record};

struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        // if self.enabled(record.metadata()) {
        println!("{} - {}", record.level(), record.args());
        // }
    }

    fn flush(&self) {}
}
use log::{LevelFilter, SetLoggerError};

static LOGGER: SimpleLogger = SimpleLogger;

pub fn init() -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Info))
}
#[cfg(feature = "gpu")]
use akari::gpu::{pt::WavefrontPathTracer, scene::GPUScene};
#[cfg(feature = "gpu")]
fn render_gpu(
    scene: &Scene,
    output: &String,
    validation: bool,
    integrator: &mut WavefrontPathTracer,
) {
    log::info!("initializing vulkan context");
    let ctx = Context::new_compute_only(vkc::ContextCreateInfo {
        enabled_extensions: &[vkc::Extension::RayTracing, vkc::Extension::ExternalMemory],
        enable_validation: validation,
    });
    log::info!("building GPUAccel");
    let gpu_scene = GPUScene::new(&ctx, scene);
    log::info!("rendering...");
    let (film, time) = profile_fn(|| -> Film { integrator.render(&gpu_scene, &scene) });
    log::info!("took {}s", time);
    let image = film.to_rgb_image();
    image.save(output).unwrap();
}

fn main() {
    init().unwrap();
    let matches = App::new("AkariRender")
        .version("0.1.0")
        .arg(
            Arg::with_name("scene")
                .short("s")
                .long("scene")
                .value_name("SCENE")
                .required(true),
        )
        .arg(
            Arg::with_name("algorithm")
                .short("a")
                .long("algo")
                .value_name("ALGORITHM")
                .help("Render algorithm")
                .required(true),
        )
        .arg(
            Arg::with_name("accel")
                .long("as")
                .alias("accel")
                .alias("acc")
                .value_name("ACCEL")
                .help("Acceleration structure (possible values: 'bvh', 'embree')"),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("OUTPUT")
                .help("Output file"),
        )
        .arg(
            Arg::with_name("threads")
                .long("threads")
                .short("t")
                .value_name("THREADS"),
        )
        .arg(
            Arg::with_name("gpu")
                .long("gpu")
                .value_name("GPU")
                .takes_value(false),
        )
        .arg(
            Arg::with_name("debug")
                .long("debug")
                .short("d")
                .value_name("DEBUG")
                .takes_value(false),
        )
        .arg(
            Arg::with_name("ooc")
                .long("ooc")
                .value_name("OOC")
                .takes_value(false)
                .help("Out of core rendering"),
        )
        .arg(
            Arg::with_name("profiling")
                .long("prof")
                .value_name("profiling")
                .takes_value(false)
                .help("Enable profiler"),
        )
        .get_matches();
    let profiling = matches.is_present("profiling");
    akari::util::profile::enable_profiler(profiling);
    let ooc = OocOptions {
        enable_ooc: matches.is_present("ooc"),
    };
    let mut config = Config::default();
    if let Some(threads) = matches.value_of("threads") {
        config.num_threads = String::from(threads).parse().unwrap();
    }
    akari::init(config);
    let accel = if let Some(accel) = matches.value_of("accel") {
        String::from(accel)
    } else {
        if cfg!(feature = "embree") {
            String::from("embree")
        } else {
            String::from("bvh")
        }
    };
    let gpu_mode = matches.is_present("gpu");
    log::info!("rendering mode {}", if gpu_mode { "gpu" } else { "cpu" });
    let scene = if let Some(scene) = matches.value_of("scene") {
        let path = Path::new(scene);
        api::load_scene::<LocalFileResolver>(path, gpu_mode, accel.as_str(), ooc)
    } else {
        log::error!("no filed provided");
        exit(1);
    };
    if scene.lights.is_empty() {
        log::error!("scene has no light!");
        exit(1);
    }
    let output = if let Some(output) = matches.value_of("output") {
        String::from(output)
    } else {
        String::from("out.png")
    };
    if !gpu_mode {
        let mut integrator = if let Some(algorithm) = matches.value_of("algorithm") {
            let path = Path::new(algorithm);
            api::load_integrator(path)
        } else {
            log::error!("no filed provided");
            exit(1);
        };
        log::info!("acceleration structure: {}", accel);
        log::info!("rendering with {} threads", rayon::current_num_threads());
        let (film, time) = profile_fn(|| -> Film { integrator.as_mut().render(&scene) });
        log::info!("took {}s", time);
        log::info!(
            "traced {} rays, average {}M rays/s",
            scene.ray_counter.load(std::sync::atomic::Ordering::Relaxed),
            scene.ray_counter.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / time,
        );
        if profiling {
            akari::util::profile::print_stats();
        }
        if output.ends_with(".exr") {
            film.write_exr(&output);
        } else {
            let image = film.to_rgb_image();
            image.save(output).unwrap();
        }
    } else {
        #[cfg(feature = "gpu")]
        {
            let mut integrator = if let Some(algorithm) = matches.value_of("algorithm") {
                let path = Path::new(algorithm);
                api::load_gpu_integrator(path)
            } else {
                log::error!("no filed provided");
                exit(1);
            };
            render_gpu(
                &scene,
                &output,
                matches.is_present("debug"),
                &mut integrator,
            );
        }
        #[cfg(not(feature = "gpu"))]
        {
            println!("gpu rendering is not enabled");
            exit(1);
        }
    }
}