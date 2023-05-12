use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let out = cmake::Config::new("ext/assimp")
        .no_build_target(true)
        .build();
    dbg!(out.display());
    println!("cargo:rustc-link-search=native={}/build/bin", out.display());
    match env::var("PROFILE") {
        Ok(x) => {
            if x == "release" {
                println!("cargo:rustc-link-lib=dylib=assimp");
            } else if x == "debug" {
                println!("cargo:rustc-link-lib=dylib=assimpd");
            }
        }
        _ => unreachable!(),
    }
}
