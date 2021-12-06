use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::slice::Iter;
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn read_mat(iter: &mut Iter<f64>) -> Vec<f64> {
    let mut data = vec![];
    let a = *iter.next().unwrap() as usize;
    let b = *iter.next().unwrap() as usize;
    assert!(a == 64 && b == 64);
    for _ in 0..a * b {
        data.push(*iter.next().unwrap());
    }
    data
}
fn write_ltc_fit() {
    let lines = read_lines("src/ltc/ltc.mat").unwrap();
    let numbers: Vec<_> = lines
        .into_iter()
        .filter(|line| line.is_ok())
        .map(|x| x.unwrap())
        .filter(|line| !line.starts_with("#"))
        .flat_map(|line| {
            line.split(" ")
                .map(|x| String::from(x))
                .collect::<Vec<String>>()
        })
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<f64>().unwrap())
        .collect();
    let mut iter = numbers.iter();
    let tab_amp = read_mat(&mut iter);
    let tab00 = read_mat(&mut iter);
    let tab10 = read_mat(&mut iter);
    let tab20 = read_mat(&mut iter);
    let tab01 = read_mat(&mut iter);
    let tab11 = read_mat(&mut iter);
    let tab21 = read_mat(&mut iter);
    let tab02 = read_mat(&mut iter);
    let tab12 = read_mat(&mut iter);
    let tab22 = read_mat(&mut iter);
    assert!(iter.next().is_none());
    let mut out = String::new();
    out += "use super::GgxLtcfit;\n";
    out += "#[allow(dead_code)]\npub const GGX_LTC_FIT:GgxLtcfit = GgxLtcfit{\nmat:[";
    for i in 0..64 * 64 {
        let m00 = tab00[i];
        let m10 = tab10[i];
        let m20 = tab20[i];
        let m01 = tab01[i];
        let m11 = tab11[i];
        let m21 = tab21[i];
        let m02 = tab02[i];
        let m12 = tab12[i];
        let m22 = tab22[i];

        out += &format!(
            "  [{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}],\n",
            m00, m10, m20, m01, m11, m21, m02, m12, m22
        );

        // out += &format!(
        //     "  [{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}],\n",
        //     m00, m01, m02, m10, m11, m12, m20, m21, m22
        // );
    }
    out += "],\n  amp:[\n";
    for i in 0..64 * 64 {
        out += &format!("{:.6},", tab_amp[i]);
    }
    out += "]};\n";
    std::fs::write("src/ltc/fit.rs", out).unwrap();
}
#[cfg(feature = "gpu")]
use shaderc::{
    CompileOptions, Compiler, IncludeCallbackResult, ResolvedInclude,
    ShaderKind,
};

#[derive(Debug, Clone, Copy)]
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
    println!("cargo:rerun-if-changed=src/ltc/ltc.mat");

    println!("cargo:rerun-if-changed=src/gpu/shaders");
    write_ltc_fit();

    let nrc_opt = Options { nrc: true };
    let pt_opt = Options { nrc: false };

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
