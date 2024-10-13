use bindgen::Builder;

fn main() {
    let bindings = Builder::default()
        .header("../../cpp_extension/rust-api.h")
        .clang_args(vec![
            "-x",
            "c++",
            "-std=c++23",
        ])
        .dynamic_library_name("AkariCppExt")
        .allowlist_function(".*embree_api_create")
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");
}