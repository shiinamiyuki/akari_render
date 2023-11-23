use clap::{arg, builder::BoolishValueParser, Arg, ArgAction, Command};
use std::{env::current_exe, path::PathBuf, process};

fn find_blender() -> PathBuf {
    let blender_major_version = 4;
    let blender_minor_version = 0;
    match std::env::var("BLENDER_PATH") {
        Ok(path) if !path.is_empty() => PathBuf::from(path),
        _ => {
            if cfg!(target_os = "windows") {
                let blender_path = PathBuf::from(format!(
                    "C:\\Program Files\\Blender Foundation\\Blender {}.{}\\blender.exe",
                    blender_major_version, blender_minor_version
                ));
                if blender_path.exists() {
                    blender_path
                } else {
                    eprintln!(
                        "Blender {}.{} not found. Please set BLENDER_PATH environment variable.",
                        blender_major_version, blender_minor_version
                    );
                    process::exit(1);
                }
            } else {
                todo!()
            }
        }
    }
}

fn main() {
    let mut cmd = Command::new("akari_import")
        .arg(arg!(-i --in <BLEND_FILE> "Import .blend file"))
        .arg(arg!(-o --out <DIR> "Output directory"))
        .arg(
            Arg::new("force")
                .long("force")
                .short('f')
                .action(ArgAction::SetTrue)
                .value_parser(BoolishValueParser::new()),
        );
    let help = cmd.render_help();
    let matches = cmd.get_matches();
    let input = matches.get_one::<String>("in").unwrap_or_else(|| {
        println!("{}", help);
        process::exit(1);
    });
    let output = matches.get_one::<String>("out").unwrap_or_else(|| {
        println!("{}", help);
        process::exit(1);
    });
    std::fs::create_dir_all(&output).unwrap();
    let force = matches.get_one::<bool>("force").copied().unwrap_or(false);

    let input_abs_path = std::fs::canonicalize(input).unwrap();
    let exporter_py = include_str!("../../python/exporter.py");

    {
        // write exporter.py to output directory
        std::fs::write(format!("{}/__exporter.py", output), exporter_py).unwrap();
    }

    let blender_path = find_blender();
    let mut cmd = process::Command::new(blender_path);
    cmd.arg("-b")
        .arg(input)
        .arg("-P")
        .arg(format!("{}/__exporter.py", output))
        .arg("--")
        .arg(input_abs_path)
        .arg(output);
    let exe_path = current_exe().unwrap();
    let exe_dir = exe_path.parent().unwrap();
    let akari_blender_dll_path = if cfg!(target_os = "windows") {
        exe_dir.join("akari_api.dll")
    } else {
        exe_dir.join("libakari_api.so")
    };
    cmd.env("AKARI_API_PATH", akari_blender_dll_path);
    if force {
        cmd.arg("--force");
    }
    let ec = cmd.spawn().unwrap().wait().unwrap();
    if !ec.success() {
        eprintln!("Blender exited with error code {}", ec);
        process::exit(1);
    }
    {
        std::fs::remove_file(format!("{}/__exporter.py", output)).unwrap();
    }
}
