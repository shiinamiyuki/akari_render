use std::path::PathBuf;

use bindgen::Builder;

fn fix_windows_path(path: PathBuf) -> PathBuf {
    if cfg!(target_os = "windows") {
        let path_str = path
            .to_str()
            .unwrap()
            .replace(r"\\?\", "")
            .replace(r"\", "/");
        PathBuf::from(path_str)
    } else {
        path
    }
}
fn copy_if_different(src: &str, dst: &str) {
    if let Ok(src_meta) = std::fs::metadata(src) {
        if let Ok(dst_meta) = std::fs::metadata(dst) {
            if src_meta.len() == dst_meta.len() {
                if let Ok(src_time) = src_meta.modified() {
                    if let Ok(dst_time) = dst_meta.modified() {
                        if src_time <= dst_time {
                            return;
                        }
                    }
                }
                return;
            }
        }
    }
    std::fs::copy(src, dst).unwrap();
}
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../cpp_extension/include");
    println!("cargo:rerun-if-changed=../../cpp_extension/src");
    println!("cargo:rerun-if-changed=../../cpp_extension/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../../cpp_extension/vcpkg.json");
    println!("cargo:rerun-if-changed=../../cmake-build-embree/build/CMakeCache.txt");
    fn add_ccache_args(config: &mut cmake::Config) {
        let ccaches = ["ccache", "sccache"];
        for ccache in ccaches.iter() {
            if let Ok(ccache_path) = std::process::Command::new("which").arg(ccache).output() {
                if ccache_path.status.success() {
                    config.define("CMAKE_C_COMPILER_LAUNCHER", ccache);
                    config.define("CMAKE_CXX_COMPILER_LAUNCHER", ccache);
                    break;
                }
            }
        }
    }

    // build embree
    {
        let mut config = cmake::Config::new("../../cpp_extension/ext/embree");
        config.out_dir("../../cmake-build-embree");
        config.define("EMBREE_ISPC_SUPPORT", "OFF");
        config.define("EMBREE_TUTORIALS", "OFF");
        config.define("EMBREE_STATIC_LIB", "ON");
        config.define("EMBREE_TASKING_SYSTEM", "INTERNAL");
        config.define("CMAKE_BUILD_TYPE", "Release");
        config.define("EMBREE_GEOMETRY_QUAD", "OFF");
        config.define("EMBREE_GEOMETRY_SUBDIVISION", "OFF");
        config.define("CMAKE_EXPORT_COMPILE_COMMANDS", "ON");
        add_ccache_args(&mut config);
        config.generator("Ninja");
        if cfg!(target_os = "windows") {
            config
                .define("CMAKE_C_FLAGS", "")
                .define("CMAKE_CXX_FLAGS", "")
                .define("CMAKE_C_COMPILER", "clang")
                .define("CMAKE_CXX_COMPILER", "clang++");
        }
        config.define("CMAKE_INSTALL_PREFIX", "../../cmake-build-embree/install");
        config.build_target("install");
        config.build();
    }
    // build other stuff
    {
        let mut config = cmake::Config::new("../../cpp_extension");
        config.out_dir("../../cmake-build-cpp-ext");
        config.define("CMAKE_INSTALL_PREFIX", "../../cmake-build-cpp-ext/install");
        config.define("CMAKE_EXPORT_COMPILE_COMMANDS", "ON");
        add_ccache_args(&mut config);
        config.generator("Ninja");
        config.build_target("akari_cpp_ext");
        let mut path = fix_windows_path(config.build().canonicalize().unwrap());
        path.push("build");
        let profile = std::env::var("PROFILE").unwrap();
        let target_dir = format!("../../target/{}", profile);
        std::fs::create_dir_all(&target_dir).unwrap();
        for entry in std::fs::read_dir(&path).unwrap() {
            let entry = entry.unwrap();
            let entry_path = entry.path();
            if entry_path.extension().and_then(|s| s.to_str()) == Some("dll") {
                copy_if_different(
                    entry_path.to_str().unwrap(),
                    &format!("{}/{}", target_dir, entry_path.file_name().unwrap().to_str().unwrap()),
                );
            }
        }
    }

    let bindings = Builder::default()
        .header("../../cpp_extension/include/rust-api.h")
        .clang_args(vec!["-x", "c++", "-std=c++23"])
        .dynamic_library_name("AkariCppExt")
        .allowlist_function(".*create_.*_api")
        // .enable_cxx_namespaces()
        .newtype_enum(".*")
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("src/bindings.rs")
        .expect("Couldn't write bindings!");
}
