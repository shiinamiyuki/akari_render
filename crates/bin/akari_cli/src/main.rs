use akari::api::OocOptions;
use akari::cli::{parse_arg, parse_str_to_args};
// use akari::accel::*;
// use akari::bsdf::*;
// use akari::camera::*;
// use akari::integrator::*;
// use akari::light::*;
use akari::util::LocalFileResolver;
use akari::*;
use std::env::{args, current_exe};
use std::fmt::Display;
#[cfg(feature = "gpu")]
use vkc::Context;
// use akari::shape::*;
// use akari::*;
use std::path::Path;
use std::process::exit;
use std::str::FromStr;
// use std::sync::Arc;
// extern crate clap;
use akari::film::Film;
use akari::profile_fn;
use akari::{api, rayon};
// use clap::{App, Arg};

// #[derive(Clone, Debug)]
// struct RemoteOptions {

// }
#[derive(Debug, Default)]
struct AppOptions {
    pub num_threads: Option<usize>,
    pub log_output: Option<String>,
    pub output: Option<String>,
    pub scene: Option<String>,
    pub algorithm: Option<String>,
    pub accel: Option<String>,
    pub launch_as_remote: bool,
}

fn usage() -> String {
    let mut s = String::new();
    s.push_str(
        r"AkariRender command line interface

Usage: 
    akr-cli [OPTIONS] -s <SCENE FILE>

Options:

    Basics:
    -c, --config file       read command line arguments from <file>
                            must be the first argument when supplied
    -s, --scene file        scene file
    -r, --render file       rendering algorithm
                            must be suppied unless --resume is supplied
    -a, --as name           acceleration structure, one of ('embree', 'bvh', 'qbvh')
    -o, --output file       output file, overrides settings in <RENDEDER FILE>
    -t, --threads count     specifiy number of threads
    -q, --quiet             suppress all loggings except error
    

    Miscellaneous:
    --log-ouput file        redirects logging output to file (default is stdout)
    --display-server [port] connects to tev display on [port]
    --checkpoint file,time  stores a checkpoint file every <time> seconds
                            panics if integrator does not support checkpointing
    --resume file           resumes previous render session
                            panics if integrator does not support checkpointing
                            or session is inconsistent with previous settings

    Network rendering:
    -n, --net config        connect to remote host(s) with arguments in config file.
                            config file consists of lines of the form:
                                username@hostname[:port] args
                            ssh must be present in PATH
                            password less authentication must be enabled

    Advanced:
    --launch-as-remote      launch this instance as a remote host
                            typically used by primary node in network rendering
    
",
    );
    s
}
fn render_main(options: AppOptions) {
    let mut config = Config::default();
    if let Some(t) = options.num_threads {
        config.num_threads = t;
    }
    if options.launch_as_remote || options.log_output.is_some() {
        let mut path = current_exe().unwrap();
        path.push(options.log_output.clone().unwrap_or("log.txt".into()));
        let s = path.as_os_str().to_string_lossy().into_owned();
        config.log_output = s;
    }

    akari::init(config);
    if options.launch_as_remote {
        // we dont want anything to write to stdout
        akari::util::enable_progress_bar(false);
    }
    let accel = options
        .accel
        .clone()
        .unwrap_or(if cfg!(feature = "embree") {
            "embree".into()
        } else {
            "bvh".into()
        });
    let ooc = OocOptions { enable_ooc: false };
    let scene = if let Some(scene) = &options.scene {
        let path = Path::new(scene);
        api::load_scene::<LocalFileResolver>(path, false, accel.as_str(), ooc)
    } else {
        log::error!("no filed provided");
        exit(1);
    };
    if scene.lights.is_empty() {
        log::error!("scene has no light!");
        exit(1);
    }
    let output = options.output.clone().unwrap_or("out.png".into());
    let mut integrator = if let Some(algorithm) = &options.algorithm {
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
    // if profiling {
    //     akari::util::profile::print_stats();
    // }
    if output.ends_with(".exr") {
        film.write_exr(&output);
    } else {
        let image = film.to_rgb_image();
        image.save(output).unwrap();
    }
}
fn parse_options(args: Vec<String>) -> AppOptions {
    macro_rules! on_err {
        () => {
            |e| {
                eprintln!("{}", e);
                println!("{}", usage());
                exit(-1);
            }
        };
    }
    let mut pos = 0;
    macro_rules! parse_str {
        ($long:literal) => {
            parse_arg::<String>(&args, &mut pos, $long, None).unwrap_or_else(on_err!())
        };
        ($long:literal,$short:literal) => {
            parse_arg::<String>(&args, &mut pos, $long, Some($short)).unwrap_or_else(on_err!())
        };
    }
    macro_rules! parse_int {
        ($long:literal) => {
            parse_arg::<usize>(&args, &mut pos, $long, None).unwrap_or_else(on_err!())
        };
        ($long:literal,$short:literal) => {
            parse_arg::<usize>(&args, &mut pos, $long, Some($short)).unwrap_or_else(on_err!())
        };
    }
    if let Some(config) = parse_str!("--config", "-c") {
        let config = std::fs::read_to_string(config).unwrap();
        let args = parse_str_to_args(&config);
        return parse_options(args);
    }
    let mut options = AppOptions::default();
    while pos < args.len() {
        if let Some(scene) = parse_str!("--scene", "-s") {
            options.scene = Some(scene);
        } else if let Some(render) = parse_str!("--render", "-r") {
            options.algorithm = Some(render);
        } else if let Some(threads) = parse_int!("--threads", "-t") {
            options.num_threads = Some(threads);
        } else {
            eprintln!("unrecognized option {}", args[pos]);
            exit(-1);
        }
    }
    options
}
fn real_main(args: Vec<String>) {
    let options = parse_options(args);
    render_main(options);
}
fn main() {
    let args: Vec<String> = args().skip(1).collect();
    real_main(args);
}
// #[cfg(feature = "gpu")]
// use akari::gpu::{pt::WavefrontPathTracer, scene::GPUScene};
// #[cfg(feature = "gpu")]
// fn render_gpu(
//     scene: &Scene,
//     output: &String,
//     validation: bool,
//     integrator: &mut WavefrontPathTracer,
// ) {
//     log::info!("initializing vulkan context");
//     let ctx = Context::new_compute_only(vkc::ContextCreateInfo {
//         enabled_extensions: &[vkc::Extension::RayTracing, vkc::Extension::ExternalMemory],
//         enable_validation: validation,
//     });
//     log::info!("building GPUAccel");
//     let gpu_scene = GPUScene::new(&ctx, scene);
//     log::info!("rendering...");
//     let (film, time) = profile_fn(|| -> Film { integrator.render(&gpu_scene, &scene) });
//     log::info!("took {}s", time);
//     let image = film.to_rgb_image();
//     image.save(output).unwrap();
// }

// #[derive(Clone, Debug)]
// pub struct AppOptions {
//     pub gpu:bool,
//     pub scene:String,
//     pub algorithm:String,
//     pub accel:String,
//     pub num_threads:usize,
// }
// fn main() {
//     init().unwrap();
//     let matches = App::new("AkariRender")
//         .version("0.1.0")
//         .arg(
//             Arg::with_name("scene")
//                 .short("s")
//                 .long("scene")
//                 .value_name("SCENE")
//                 .required(true),
//         )
//         .arg(
//             Arg::with_name("algorithm")
//                 .short("a")
//                 .long("algo")
//                 .value_name("ALGORITHM")
//                 .help("Render algorithm")
//                 .required(true),
//         )
//         .arg(
//             Arg::with_name("accel")
//                 .long("as")
//                 .alias("accel")
//                 .alias("acc")
//                 .value_name("ACCEL")
//                 .help("Acceleration structure (possible values: 'bvh', 'embree')"),
//         )
//         .arg(
//             Arg::with_name("output")
//                 .short("o")
//                 .long("output")
//                 .value_name("OUTPUT")
//                 .help("Output file"),
//         )
//         .arg(
//             Arg::with_name("threads")
//                 .long("threads")
//                 .short("t")
//                 .value_name("THREADS"),
//         )
//         .arg(
//             Arg::with_name("gpu")
//                 .long("gpu")
//                 .value_name("GPU")
//                 .takes_value(false),
//         )
//         .arg(
//             Arg::with_name("debug")
//                 .long("debug")
//                 .short("d")
//                 .value_name("DEBUG")
//                 .takes_value(false),
//         )
//         .arg(
//             Arg::with_name("ooc")
//                 .long("ooc")
//                 .value_name("OOC")
//                 .takes_value(false)
//                 .help("Out of core rendering"),
//         )
//         .arg(
//             Arg::with_name("profiling")
//                 .long("prof")
//                 .value_name("profiling")
//                 .takes_value(false)
//                 .help("Enable profiler"),
//         )
//         .get_matches();
//     let profiling = matches.is_present("profiling");
//     akari::util::profile::enable_profiler(profiling);
//     let ooc = OocOptions {
//         enable_ooc: matches.is_present("ooc"),
//     };
//     let mut config = Config::default();
//     if let Some(threads) = matches.value_of("threads") {
//         config.num_threads = String::from(threads).parse().unwrap();
//     }
//     akari::init(config);
//     let accel = if let Some(accel) = matches.value_of("accel") {
//         String::from(accel)
//     } else {
//         if cfg!(feature = "embree") {
//             String::from("embree")
//         } else {
//             String::from("bvh")
//         }
//     };
//     let gpu_mode = matches.is_present("gpu");
//     log::info!("rendering mode {}", if gpu_mode { "gpu" } else { "cpu" });
//     let scene = if let Some(scene) = matches.value_of("scene") {
//         let path = Path::new(scene);
//         api::load_scene::<LocalFileResolver>(path, gpu_mode, accel.as_str(), ooc)
//     } else {
//         log::error!("no filed provided");
//         exit(1);
//     };
//     if scene.lights.is_empty() {
//         log::error!("scene has no light!");
//         exit(1);
//     }
//     let output = if let Some(output) = matches.value_of("output") {
//         String::from(output)
//     } else {
//         String::from("out.png")
//     };
//     if !gpu_mode {
//         let mut integrator = if let Some(algorithm) = matches.value_of("algorithm") {
//             let path = Path::new(algorithm);
//             api::load_integrator(path)
//         } else {
//             log::error!("no filed provided");
//             exit(1);
//         };
//         log::info!("acceleration structure: {}", accel);
//         log::info!("rendering with {} threads", rayon::current_num_threads());
//         let (film, time) = profile_fn(|| -> Film { integrator.as_mut().render(&scene) });
//         log::info!("took {}s", time);
//         log::info!(
//             "traced {} rays, average {}M rays/s",
//             scene.ray_counter.load(std::sync::atomic::Ordering::Relaxed),
//             scene.ray_counter.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / time,
//         );
//         if profiling {
//             akari::util::profile::print_stats();
//         }
//         if output.ends_with(".exr") {
//             film.write_exr(&output);
//         } else {
//             let image = film.to_rgb_image();
//             image.save(output).unwrap();
//         }
//     } else {
//         #[cfg(feature = "gpu")]
//         {
//             let mut integrator = if let Some(algorithm) = matches.value_of("algorithm") {
//                 let path = Path::new(algorithm);
//                 api::load_gpu_integrator(path)
//             } else {
//                 log::error!("no filed provided");
//                 exit(1);
//             };
//             render_gpu(
//                 &scene,
//                 &output,
//                 matches.is_present("debug"),
//                 &mut integrator,
//             );
//         }
//         #[cfg(not(feature = "gpu"))]
//         {
//             println!("gpu rendering is not enabled");
//             exit(1);
//         }
//     }
// }
