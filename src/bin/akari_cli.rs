use akari_render::{
    film::{Film, FilmColorRepr},
    integrator::{
        mcmc::{self, MCMC},
        normal::NormalVis,
        pt::PathTracer,
        Integrator,
    },
};
use clap::{arg, Arg, Command};
use luisa_compute as luisa;
use std::{env::current_exe, process::exit};
fn main() {
    luisa::init_logger();
    let mut cmd = Command::new("akari_cli")
        .allow_missing_positional(false)
        .arg(Arg::new("scene").help("Scene file to render"))
        .arg(
            arg!(-d --device <DEVICE> "Compute device. One of: cpu, cuda, dx, metal. Default: cpu"),
        )
        .arg(arg!(-o --output <FILE> "Output file name. Default: out.png"))
        .arg(arg!(--spp <SPP> "Set samples per pixel, override existing value"));
    let help = cmd.render_help();
    let matches = cmd.get_matches();
    let scene = matches.get_one::<String>("scene");
    let output = matches
        .get_one::<String>("output")
        .cloned()
        .unwrap_or("out.png".to_string());
    let spp = matches
        .get_one::<String>("spp")
        .map(|s| s.parse::<u32>().unwrap());
    if scene.is_none() {
        println!("{}", help);
        exit(1);
    }
    let device = matches
        .get_one::<String>("device")
        .cloned()
        .unwrap_or("cpu".to_string());
    let ctx = luisa::Context::new(current_exe().unwrap());
    let device = ctx.create_device(&device);
    let scene = akari_render::scene::Scene::load(device.clone(), &scene.unwrap());
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
    // {
    //     let normal_vis = NormalVis::new(device.clone(), 1);
    //     normal_vis.render(&scene, &mut film).unwrap_or_else(|e| {
    //         println!("Render failed: {:?}", e);
    //         exit(1);
    //     });
    // }
    let tic = std::time::Instant::now();
    // {
    //     let pt = PathTracer::new(device.clone(), spp.unwrap_or(1), 64, 5, true);
    //     pt.render(&scene, &mut film);
    // }
    {
        let mcmc = MCMC::new(
            device.clone(),
            spp.unwrap_or(1),
            1,
            5,
            true,
            mcmc::Method::Kelemen {
                small_sigma: 0.01,
                large_step_prob: 0.1,
            },
            100,
            100000,
        );
        mcmc.render(&scene, &mut film);
    }
    film.copy_to_rgba_image(&output_image);
    let toc = std::time::Instant::now();
    log::info!("Rendered in {:.1}ms", (toc - tic).as_secs_f64() * 1e3);

    akari_render::util::write_image(&output_image, &output);
}
