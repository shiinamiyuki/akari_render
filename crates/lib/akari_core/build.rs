#[cfg(feature = "gpu")]
use shaderc::{CompileOptions, Compiler, IncludeCallbackResult, ResolvedInclude, ShaderKind};
use std::fs::{canonicalize, create_dir};
use std::path::Path;
use std::{env, fs};
use std::{path::PathBuf, process::Command};
// fn use_this_when_cc_supports_vs22() -> PathBuf {
//     let mut config = cmake::Config::new("src/pbrt-v4-rgb2spec-opt");
//     config
//         .define("CMAKE_BUILD_TYPE", "Release")
//         .build_arg("--config")
//         .build_arg("RELEASE");
//     config.build()
// }
fn compile_rgb2spec_opt() {
    if !Path::new("src/pbrt-v4-rgb2spec-opt/build").exists() {
        create_dir("src/pbrt-v4-rgb2spec-opt/build").unwrap();
    }
    Command::new("cmake")
        .args([".."])
        .current_dir("./src/pbrt-v4-rgb2spec-opt/build")
        .output()
        .expect("cmake failed to start");
    let mut cmd = Command::new("cmake");
    cmd.args(["--build", "."]);
    if cfg!(target_os = "windows") {
        cmd.args(["--config", "Release"]);
    }
    cmd.current_dir("./src/pbrt-v4-rgb2spec-opt/build")
        .output()
        .expect("cmake failed to start");
    let exe = if cfg!(target_os = "windows") {
        PathBuf::from("src/pbrt-v4-rgb2spec-opt/build/Release/rgb2spec_opt.exe")
    } else {
        PathBuf::from("src/pbrt-v4-rgb2spec-opt/build/rgb2spec_opt")
    };
    let dst = env::var("OUT_DIR").unwrap();

    let dst = PathBuf::from(dst);
    let comps: Vec<_> = dst.components().collect();
    let dst = PathBuf::from_iter(comps[..comps.len() - 3].iter());
   
    let dst = format!(
        "{}/rgb2spec_opt{}",
        dst.display(),
        if cfg!(target_os = "windows") {
            ".exe"
        } else {
            ""
        }
    );
    println!("{}", dst);
    fs::copy(exe, dst).unwrap();
}
// fn run_rgb2spec_opt(dir: PathBuf) {
//     let mut rgb2spec = dir.clone();
//     rgb2spec.push("rgb2spec_opt");
//     if cfg!(target_os = "windows") {
//         rgb2spec.set_extension("exe");
//     }
//     let color_spaces = ["ACES2065_1", "DCI_P3", "REC2020", "sRGB"];
//     let curent_dir = current_dir().unwrap();
//     for color_space in color_spaces {
//         let mut output = curent_dir.clone();
//         output.push(format!(
//             "src/rgbspectrum_{}",
//             color_space.to_ascii_lowercase()
//         ));
//         if output.clone().into_boxed_path().exists() {
//             continue;
//         }
//         let args: Vec<OsString> = vec!["64".into(), output.into_os_string(), color_space.into()];
//         Command::new(&rgb2spec)
//             .args(args)
//             .current_dir(canonicalize(current_dir().unwrap()).unwrap())
//             .output()
//             .expect(&format!(
//                 "{} failed to start",
//                 rgb2spec.clone().into_os_string().into_string().unwrap()
//             ));
//     }
// }
fn rgb2spec() {
    println!("cargo:rerun-if-changed=src/pbrt-v4-rgb2spec-opt/rgb2spec_opt.cpp");
    println!("cargo:rerun-if-changed=src/pbrt-v4-rgb2spec-opt/CMakeLists.txt");
    compile_rgb2spec_opt();
}

// #[derive(Debug, Clone, Copy)]
// #[allow(dead_code)]
// struct Options {
//     nrc: bool,
// }

// #[cfg(feature = "gpu")]
// fn compile_shader_imp(path: &String, shader_kind: ShaderKind, output: &String, options2: Options) {
//     use std::{io::Write, process::exit};
//     println!("{}", path);
//     let mut compiler = Compiler::new().unwrap();
//     let source = std::fs::read_to_string(path).unwrap();
//     let mut options = CompileOptions::new().unwrap();
//     options.set_include_callback(
//         |name, _include_type, _src, _depth| -> IncludeCallbackResult {
//             let filename = format!("src/gpu/shaders/{}", name);
//             let content = std::fs::read_to_string(&filename).unwrap_or_else(|_| {
//                 panic!("failed to resolved include {}", name);
//             });
//             Ok(ResolvedInclude {
//                 content,
//                 resolved_name: filename,
//             })
//         },
//     );
//     if options2.nrc {
//         options.add_macro_definition("ENABLE_NRC", None);
//     }
//     let artifact = compiler.compile_into_spirv(&source, shader_kind, path, "main", Some(&options));
//     let spirv = match artifact {
//         Ok(spirv) => spirv,
//         Err(err) => {
//             log::error!("Shader Compilation Failure:\nFile:{}\n{}", path, err);
//             exit(1);
//         }
//     };
//     let mut file = File::create(output).unwrap();
//     file.write_all(spirv.as_binary_u8()).unwrap();
// }

// #[cfg(feature = "gpu")]
// fn compile_shader(path: &str, shader_kind: ShaderKind, output: &str, options: Options) {
//     if !Path::new("src/gpu/spv").exists() {
//         std::fs::create_dir("src/gpu/spv").unwrap();
//     }
//     compile_shader_imp(
//         &format!("src/gpu/shaders/{}", path),
//         shader_kind,
//         &format!("src/gpu/spv/{}", output),
//         options,
//     )
// }

fn main() {
    rgb2spec();

    // let nrc_opt = Options { nrc: true };
    // let pt_opt = Options { nrc: false };

    // let _ = nrc_opt;
    // let _ = pt_opt;

    // #[cfg(feature = "gpu")]
    // {
    //     println!("cargo:rerun-if-changed=src/gpu/shaders");
    //     compile_shader(
    //         "closest.rgen.glsl",
    //         ShaderKind::RayGeneration,
    //         "closest.rgen.spv",
    //         pt_opt,
    //     );

    //     compile_shader(
    //         "closest.rchit.glsl",
    //         ShaderKind::ClosestHit,
    //         "closest.rchit.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "closest.miss.glsl",
    //         ShaderKind::Miss,
    //         "closest.miss.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "shadow.rgen.glsl",
    //         ShaderKind::RayGeneration,
    //         "shadow.rgen.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "shadow.rgen.glsl",
    //         ShaderKind::RayGeneration,
    //         "nrc_shadow.rgen.spv",
    //         nrc_opt,
    //     );
    //     compile_shader(
    //         "shadow.rchit.glsl",
    //         ShaderKind::ClosestHit,
    //         "shadow.rchit.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "shadow.miss.glsl",
    //         ShaderKind::Miss,
    //         "shadow.miss.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "perspective_camera.glsl",
    //         ShaderKind::Compute,
    //         "perspective_camera.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "reset_queue.glsl",
    //         ShaderKind::Compute,
    //         "reset_queue.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "reset_ray_queue.glsl",
    //         ShaderKind::Compute,
    //         "reset_ray_queue.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "material_eval.glsl",
    //         ShaderKind::Compute,
    //         "material_eval.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "material_eval.glsl",
    //         ShaderKind::Compute,
    //         "nrc_material_eval.spv",
    //         nrc_opt,
    //     );
    //     compile_shader(
    //         "init_path_states.glsl",
    //         ShaderKind::Compute,
    //         "init_path_states.spv",
    //         pt_opt,
    //     );
    //     compile_shader(
    //         "splat_film.glsl",
    //         ShaderKind::Compute,
    //         "splat_film.spv",
    //         pt_opt,
    //     );
    // }
}
