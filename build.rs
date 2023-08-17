use std::env;
use std::fs::{self, create_dir, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
fn srgb_to_linear1(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        (((s + 0.055) / 1.055) as f32).powf(2.4)
    }
}
fn linear_to_srgb1(l: f32) -> f32 {
    if l <= 0.0031308 {
        l * 12.92
    } else {
        l.powf(1.0 / 2.4) * 1.055 - 0.055
    }
}
fn gen_rgb_lut() {
    let mut f = File::create("src/data/rgb8.rs").unwrap();
    writeln!(&mut f, "pub const LINEAR_TO_SRGB:[f32;256]=[").unwrap();
    for i in 0..256 {
        let x = i as f32 / 255.0;
        write!(&mut f, "{}f32,", linear_to_srgb1(x)).unwrap();
    }
    write!(&mut f, "];\n").unwrap();
    writeln!(&mut f, "pub const SRGB_TO_LINEAR:[f32;256]=[").unwrap();
    for i in 0..256 {
        let x = i as f32 / 255.0;
        write!(&mut f, "{}f32,", srgb_to_linear1(x)).unwrap();
    }
    write!(&mut f, "];\n").unwrap();
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp_ext");
    // gen_rgb2spec();
    let out = cmake::Config::new("cpp_ext")
        .generator("Ninja")
        .define("CMAKE_BUILD_TYPE", "Release")
        .no_build_target(true)
        .build();
    dbg!(out.display());
    println!("cargo:rustc-link-search=native={}/build", out.display());
    println!("cargo:rustc-link-lib=static=akari_cpp_ext");
}
