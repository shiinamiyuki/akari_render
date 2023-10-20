fn main() {
    println!("cargo:rerun-if-changed=blender_src_path");
    let blender_src_path = std::fs::read("blender_src_path.txt")
        .expect("blender_src_path.txt file not found")
        .into_iter()
        .map(|b| b as char)
        .collect::<String>();
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cpp_ext");
    let out = cmake::Config::new("cpp_ext")
        .generator("Ninja")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BLENDER_SRC_PATH", blender_src_path)
        .no_build_target(true)
        .build();
    dbg!(out.display());
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
    println!("cargo:rustc-link-search=native={}/build", out.display());
    println!("cargo:rustc-link-lib=static=akari_blender_cpp_ext");
}
