use akari_common::serde_json;
use akari_render::clap;
use akari_render::{
    gui::DisplayWindow,
    integrator::{render, RenderSession, RenderTask},
    luisa,
};
use clap::{arg, builder::BoolishValueParser, Arg, ArgAction, Command};
use std::{env::current_exe, fs::File, process::exit};

fn main() {
    let mut cmd = Command::new("akari_cli")
        .arg(arg!(-s --scene <SCENE> "Scene file to render"))
        .arg(arg!(-m --method <METHOD> "Render method config file"))
        .arg(
            arg!(-d --device <DEVICE> "Compute device. One of: cpu, cuda, dx, metal. Default: cpu"),
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .short('v')
                .action(ArgAction::SetTrue)
                .value_parser(BoolishValueParser::new()),
        )
        .arg(
            Arg::new("save-intermediate")
                .long("save-intermediate")
                .action(ArgAction::SetTrue)
                .value_parser(BoolishValueParser::new()),
        )
        .arg(
            Arg::new("gui")
                .long("gui")
                .action(ArgAction::SetTrue)
                .value_parser(BoolishValueParser::new()),
        )
        .arg(Arg::new("save-stats").long("save-stats"));
    let help = cmd.render_help();
    let matches = cmd.get_matches();
    let scene = matches.get_one::<String>("scene");
    let method = matches.get_one::<String>("method");
    let save_intermediate = matches.get_one::<bool>("save-intermediate").copied();
    let session = matches.get_one::<String>("save-stats").cloned();
    let gui = matches.get_one::<bool>("gui").copied().unwrap_or(false);
    if scene.is_none() {
        println!("{}", help);
        exit(1);
    }
    if method.is_none() {
        println!("{}", help);
        exit(1);
    }
    let device = matches
        .get_one::<String>("device")
        .cloned()
        .unwrap_or("cpu".to_string());
    let verbose = matches.get_one::<bool>("verbose").copied();
    if let Some(true) = verbose {
        luisa::init_logger_verbose();
    } else {
        luisa::init_logger();
    }
    let ctx = luisa::Context::new(current_exe().unwrap());
    let device = ctx.create_device(&device);
    let method = method.unwrap();
    let task: RenderTask = {
        let file = File::open(method).unwrap();
        serde_json::from_reader(file).unwrap()
    };
    let scene = akari_render::load::load_from_path(device.clone(), &scene.unwrap());
    let window = if gui {
        std::env::set_var("WINIT_UNIX_BACKEND", "x11");
        let window = DisplayWindow::new(
            &device,
            scene.camera.resolution().x,
            scene.camera.resolution().y,
        );
        Some(window)
    } else {
        None
    };
    let session = RenderSession {
        save_intermediate: save_intermediate.unwrap_or(false),
        name: session.clone().unwrap_or_else(|| String::from("default")),
        save_stats: session.is_some(),
        display: window.as_ref().map(|w| w.channel()),
    };
    let render_thread = std::thread::spawn(move || {
        render(device.clone(), scene, &task, session);
    });

    if let Some(window) = window {
        window.show();
    } else {
        render_thread.join().unwrap();
    }
}
