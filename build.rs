#[cfg(feature = "gpu")]
use shaderc::{
    CompileOptions, Compiler, IncludeCallbackResult, ResolvedInclude,
    ShaderKind,
};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct Options {
    nrc: bool,
}

#[cfg(feature = "gpu")]
fn compile_shader_imp(path: &String, shader_kind: ShaderKind, output: &String, options2: Options) {
    use std::{io::Write, process::exit};
    println!("{}", path);
    let mut compiler = Compiler::new().unwrap();
    let source = std::fs::read_to_string(path).unwrap();
    let mut options = CompileOptions::new().unwrap();
    options.set_include_callback(
        |name, _include_type, _src, _depth| -> IncludeCallbackResult {
            let filename = format!("src/gpu/shaders/{}", name);
            let content = std::fs::read_to_string(&filename).unwrap_or_else(|_| {
                panic!("failed to resolved include {}", name);
            });
            Ok(ResolvedInclude {
                content,
                resolved_name: filename,
            })
        },
    );
    if options2.nrc {
        options.add_macro_definition("ENABLE_NRC", None);
    }
    let artifact = compiler.compile_into_spirv(&source, shader_kind, path, "main", Some(&options));
    let spirv = match artifact {
        Ok(spirv) => spirv,
        Err(err) => {
            eprintln!("Shader Compilation Failure:\nFile:{}\n{}", path, err);
            exit(1);
        }
    };
    let mut file = File::create(output).unwrap();
    file.write_all(spirv.as_binary_u8()).unwrap();
}



#[cfg(feature = "gpu")]
fn compile_shader(path: &str, shader_kind: ShaderKind, output: &str, options: Options) {
    if !Path::new("src/gpu/spv").exists() {
        std::fs::create_dir("src/gpu/spv").unwrap();
    }
    compile_shader_imp(
        &format!("src/gpu/shaders/{}", path),
        shader_kind,
        &format!("src/gpu/spv/{}", output),
        options,
    )
}


fn main() {
    println!("cargo:rerun-if-changed=src/gpu/shaders");

    let nrc_opt = Options { nrc: true };
    let pt_opt = Options { nrc: false };

    let _ = nrc_opt;
    let _ = pt_opt;

#   [cfg(feature = "gpu")]
    {
        compile_shader(
            "closest.rgen.glsl",
            ShaderKind::RayGeneration,
            "closest.rgen.spv",
            pt_opt,
        );

        compile_shader(
            "closest.rchit.glsl",
            ShaderKind::ClosestHit,
            "closest.rchit.spv",
            pt_opt,
        );
        compile_shader(
            "closest.miss.glsl",
            ShaderKind::Miss,
            "closest.miss.spv",
            pt_opt,
        );
        compile_shader(
            "shadow.rgen.glsl",
            ShaderKind::RayGeneration,
            "shadow.rgen.spv",
            pt_opt,
        );
        compile_shader(
            "shadow.rgen.glsl",
            ShaderKind::RayGeneration,
            "nrc_shadow.rgen.spv",
            nrc_opt,
        );
        compile_shader(
            "shadow.rchit.glsl",
            ShaderKind::ClosestHit,
            "shadow.rchit.spv",
            pt_opt,
        );
        compile_shader(
            "shadow.miss.glsl",
            ShaderKind::Miss,
            "shadow.miss.spv",
            pt_opt,
        );
        compile_shader(
            "perspective_camera.glsl",
            ShaderKind::Compute,
            "perspective_camera.spv",
            pt_opt,
        );
        compile_shader(
            "reset_queue.glsl",
            ShaderKind::Compute,
            "reset_queue.spv",
            pt_opt,
        );
        compile_shader(
            "reset_ray_queue.glsl",
            ShaderKind::Compute,
            "reset_ray_queue.spv",
            pt_opt,
        );
        compile_shader(
            "material_eval.glsl",
            ShaderKind::Compute,
            "material_eval.spv",
            pt_opt,
        );
        compile_shader(
            "material_eval.glsl",
            ShaderKind::Compute,
            "nrc_material_eval.spv",
            nrc_opt,
        );
        compile_shader(
            "init_path_states.glsl",
            ShaderKind::Compute,
            "init_path_states.spv",
            pt_opt,
        );
        compile_shader(
            "splat_film.glsl",
            ShaderKind::Compute,
            "splat_film.spv",
            pt_opt,
        );
    }
}
