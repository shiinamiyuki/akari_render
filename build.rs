use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // let out = cmake::Config::new("cpp_ext")
    //     .no_build_target(true)
    //     .build();
    // dbg!(out.display());
    // println!("cargo:rustc-link-search=native={}/build", out.display());
    // println!("cargo:rustc-link-lib=static=akari_cpp_ext");
}
