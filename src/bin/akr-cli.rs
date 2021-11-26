// use akari::accel::*;
// use akari::bsdf::*;
// use akari::camera::*;
// use akari::integrator::*;
// use akari::light::*;
use akari::scene::Scene;
use vkc::Context;
// use akari::shape::*;
// use akari::*;
use std::path::Path;
use std::process::exit;
// use std::sync::Arc;
extern crate clap;
use akari::api;
use akari::film::Film;
use akari::profile;
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
    let (film, time) = profile(|| -> Film { integrator.render(&gpu_scene, &scene) });
    log::info!("took {}s", time);
    let image = film.to_rgb_image();
    image.save(output).unwrap();
}

#[cfg(not(feature = "gpu"))]

fn render_gpu(scene: &Scene) {
    println!("gpu rendering is not enabled");
    exit(1);
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
                .value_name("ACCEL")
                .help("Acceleration structure (possible values: 'bvh', 'embree')")
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("OUTPUT")
                .help("output file"),
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
        .get_matches();
    if let Some(threads) = matches.value_of("threads") {
        let threads: usize = String::from(threads).parse().unwrap();
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }
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
        api::load_scene(path, gpu_mode, accel.as_str())
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
        log::info!("Acceleration Structure: {}", accel);
        log::info!("Rendering with {} threads", rayon::current_num_threads());
        let (film, time) = profile(|| -> Film { integrator.as_mut().render(&scene) });
        log::info!("Took {}s", time);
        let image = film.to_rgb_image();
        image.save(output).unwrap();
    } else {
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
}
// fn main() {
//     // rayon::ThreadPoolBuilder::new()
//     //     .num_threads(1)
//     //     .build_global()
//     //     .unwrap();
//     let white = Arc::new(DiffuseBsdf {
//         color: Spectrum::one(),
//     });
//     let red = Arc::new(DiffuseBsdf {
//         color: Spectrum::from_srgb(&vec3(0.75, 0.25, 0.25)),
//     });
//     let green = Arc::new(DiffuseBsdf {
//         color: Spectrum::from_srgb(&vec3(0.25, 0.75, 0.25)),
//     });
//     let shape = {
//         let mut shapes: Vec<Arc<dyn Shape>> = vec![];
//         shapes.push(Arc::new(Sphere {
//             center: vec3(0.0, 0.0, -4.0),
//             radius: 1.0,
//             bsdf: white.clone(),
//         }));
//         shapes.push(Arc::new(Sphere {
//             center: vec3(10000.0 + 4.0, 0.0, -0.0),
//             radius: 10000.0,
//             bsdf: red.clone(),
//         }));
//         shapes.push(Arc::new(Sphere {
//             center: vec3(-10000.0 - 4.0, 0.0, -0.0),
//             radius: 10000.0,
//             bsdf: green.clone(),
//         }));
//         shapes.push(Arc::new(Sphere {
//             center: vec3(0.0, -10000.0 - 1.0, -0.0),
//             radius: 10000.0,
//             bsdf: white.clone(),
//         }));
//         shapes.push(Arc::new(Sphere {
//             center: vec3(0.0, 10000.0 + 6.0, -0.0),
//             radius: 10000.0,
//             bsdf: white.clone(),
//         }));
//         shapes.push(Arc::new(Sphere {
//             center: vec3(0.0, 0.0, -10015.0),
//             radius: 10000.0,
//             bsdf: white.clone(),
//         }));
//         Arc::new(Aggregate::new(shapes))
//     };
//     let camera = {
//         let m = glm::translate(&glm::identity(), &vec3(0.0, 0.4, 0.0));
//         Arc::new(PerspectiveCamera::new(
//             &uvec2(1024, 1024),
//             &Transform::from_matrix(&m),
//             (80.0 as Float).to_radians(),
//         ))
//     };
//     let lights: Vec<Arc<dyn Light>> = vec![Arc::new(PointLight {
//         emission: Spectrum::one() * 40.0,
//         position: vec3(0.3, 4.0, 0.0),
//     })];
//     let scene = Scene {
//         shape,
//         camera,
//         lights: lights.clone(),
//         light_distr: Arc::new(UniformLightDistribution::new(lights.clone())),
//     };
//     // let mut integrator = PathTracer {
//     //     spp: 16,
//     //     max_depth: 3,
//     // };
//     let mut integrator = nrc::CachedPathTracer {
//         spp: 16,
//         training_iters: 1024,
//         max_depth: 3,
//         ..Default::default()
//     };
//     // let mut integrator = Bdpt {
//     //     spp: 32,
//     //     max_depth: 3,
//     //     debug: false,
//     // };
//     // let mut integrator = Sppm {
//     //     initial_radius: 0.1,
//     //     iterations: 64,
//     //     max_depth: 5,
//     //     n_photons: 100000,
//     // };
//     let film = integrator.render(&scene);
//     let image = film.to_rgb_image();
//     image.save("out-nrc.png").unwrap();
// }
