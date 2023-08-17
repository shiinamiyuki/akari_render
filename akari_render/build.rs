use std::env;
use std::fs::{self, create_dir, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

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
