fn main() {
    let out = cmake::Config::new("ext/assimp")
        .no_build_target(true)
        .build();
    dbg!(out.display());
    println!("cargo:rustc-link-search=native={}/build/bin", out.display());
    println!("cargo:rustc-link-lib=dylib=assimpd");
}
