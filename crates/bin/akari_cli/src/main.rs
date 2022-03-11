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
    num_threads: Option<usize>,
    log_output: Option<String>,
    output: Option<String>,
    scene: Option<String>,
    algorithm: Option<String>,
    accel: Option<String>,
    listen: Option<u32>,
    remote: Vec<(String, u32)>,
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
    -n, --net host[:port]   connect to remote host(s)
    -l, --listen port       run as server mode, listen for connection at <port>

    Advanced:
    --launch-as-remote      launch this instance as a remote host
                            typically used by primary node in network rendering
    
",
    );
    s
}
fn default_accel()->String {
    if cfg!(feature = "embree") {
        "embree".into()
    } else {
        "bvh".into()
    }
}
fn server_main(options: AppOptions) {}
fn render_main(options: AppOptions) {
    let mut config = Config::default();
    if let Some(t) = options.num_threads {
        config.num_threads = t;
    }
    if options.log_output.is_some() {
        let mut path = current_exe().unwrap();
        path.push(options.log_output.clone().unwrap_or("log.txt".into()));
        let s = path.as_os_str().to_string_lossy().into_owned();
        config.log_output = s;
    }

    akari::init(config);
    let accel = options
        .accel
        .clone()
        .unwrap_or(default_accel());
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
    if options.listen.is_some() {
        server_main(options);
    } else {
        render_main(options);
    }
}

fn main() {
    let args: Vec<String> = args().skip(1).collect();
    real_main(args);
}
