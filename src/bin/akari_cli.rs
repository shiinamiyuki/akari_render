use akari_render::integrator::{render, RenderOptions, RenderTask};
use clap::{arg, builder::BoolishValueParser, Arg, ArgAction, Command};
use luisa_compute as luisa;
use std::{env::current_exe, fs::File, process::exit};
fn main() {
    luisa::init_logger();
    let mut cmd = Command::new("akari_cli")
        .arg(arg!(-s --scene <SCENE> "Scene file to render"))
        .arg(arg!(-m --method <METHOD> "Render method config file"))
        .arg(
            arg!(-d --device <DEVICE> "Compute device. One of: cpu, cuda, dx, metal. Default: cpu"),
        )
        .arg(
            Arg::new("save-intermediate")
                .long("save-intermediate")
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
    let ctx = luisa::Context::new(current_exe().unwrap());
    let device = ctx.create_device(&device);
    let method = method.unwrap();
    let task: RenderTask = {
        let file = File::open(method).unwrap();
        serde_json::from_reader(file).unwrap()
    };
    let scene = akari_render::scene::Scene::load_from_path(device.clone(), &scene.unwrap());
    let options = RenderOptions {
        save_intermediate: save_intermediate.unwrap_or(false),
        session: session.clone().unwrap_or_else(|| String::from("default")),
        save_stats: session.is_some(),
    };
    render(device, scene, &task, options);
}
