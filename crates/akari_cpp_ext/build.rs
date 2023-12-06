fn main() {
    println!("cargo:rerun-if-changed=../../blender_src_path.txt");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=./cpp_ext");
    let out = cmake::Config::new("./cpp_ext")
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
    bindgen::builder()
        .header("cpp_ext/akari_cpp_ext.h")
        .clang_args(&["-x", "c++", "-std=c++20"])
        .allowlist_file("cpp_ext/akari_cpp_ext.h")
        .enable_cxx_namespaces()
        .generate()
        .unwrap()
        .write_to_file("src/binding.rs")
        .unwrap();
}
