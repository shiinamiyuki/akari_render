
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp_ext");
    let out = cmake::Config::new("cpp_ext")
        .generator("Ninja")
        .define("CMAKE_BUILD_TYPE", "Release")
        .no_build_target(true)
        .build();
    dbg!(out.display());
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
    println!("cargo:rustc-link-search=native={}/build", out.display());
    println!("cargo:rustc-link-lib=static=akari_cpp_ext");
}
