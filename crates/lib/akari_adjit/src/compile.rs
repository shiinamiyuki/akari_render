use std::{
    env::current_exe,
    fs::{self, canonicalize},
    io::{Result, Write},
    path::{Path, PathBuf},
    process::Command,
};

const CMAKE_TEMPLATE: &'static str = r#"
cmake_minimum_required (VERSION 3.12)
project ({{TARGET}} LANGUAGES CXX C)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
	#set(CMAKE_CXX_FLAGS /std:c++17 /MP /arch:AVX2 /WX)
else()
    set(CMAKE_CXX_FLAGS -lstdc++ ${CMAKE_CXX_FLAGS})
endif()

add_library({{TARGET}} SHARED source.cpp)
"#;
fn cmake_source(target: &str) -> String {
    let s = String::from(CMAKE_TEMPLATE);
    let s = s.replace("{{TARGET}}", target.into());
    s
}
fn canonicalize_and_fix_windows_path(path: PathBuf) -> Result<PathBuf> {
    let path = canonicalize(path)?;
    let mut s: String = path.to_str().unwrap().into();
    if s.starts_with(r"\\?\") {
        // s(r"\\?\".len());
        s = s[r"\\?\".len()..].into();
    }
    Ok(PathBuf::from(s))
}
fn write_if_changed(path: &String, content: &str) -> Result<bool> {
    let path = PathBuf::from(path);
    let old = if path.exists() {
        fs::read_to_string(&path).map_err(|e| {
            eprintln!("read {} failed", path.display());
            e
        })?
    } else {
        "".into()
    };
    if old != content {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&path)
            .map_err(|e| {
                eprintln!("create {} failed", path.display());
                e
            })?;
        write!(&mut file, "{}", content)?;
        Ok(true)
    } else {
        Ok(false)
    }
}
pub fn compile(source: &str, target: &str) -> Result<PathBuf> {
    let mut source = String::from(source);
    let header = r#"
    #include <stdint.h>
    typedef uint32_t u32;
    typedef uint64_t u64;
    typedef float f32;
    typedef double f64;
    typedef int32_t i32;
    typedef uint32_t u32;
    typedef size_t usize;
    typedef ptrdiff_t isize;
    "#;
    source = format!("{}{}", header, source);
    if cfg!(target_os = "windows") {
        source = source.replace("AKR_JIT_DLL_EXPORT", "__declspec( dllexport )");
    } else {
        source = source.replace("AKR_JIT_DLL_EXPORT", "");
    }
    let self_path = current_exe().map_err(|e| {
        eprintln!("current_exe() failed");
        e
    })?;
    let self_path: PathBuf = canonicalize_and_fix_windows_path(self_path)?
        .parent()
        .unwrap()
        .into();
    let mut cmake_path = self_path.clone();
    cmake_path.push(".jit/");
    cmake_path.push(format!("{}/", target));
    let build_dir = cmake_path.clone();
    // build_dir.push("build/");
    if !build_dir.exists() {
        fs::create_dir_all(&build_dir).map_err(|e| {
            eprintln!("fs::create_dir_all({}) failed", build_dir.display());
            e
        })?;
    }
    let source_file = format!("{}/source.c", cmake_path.display());
    write_if_changed(&source_file, &source)?;
    let target_lib = if cfg!(target_os = "windows") {
        format!("{}.dll", target)
    } else {
        format!("{}.so", target)
    };
    match Command::new("clang")
        .args(["-shared", "-O3", "-o", &target_lib, &source_file])
        .current_dir(&build_dir)
        .spawn()
        .expect("clang failed to start")
        .wait_with_output()
        .expect("clang failed")
    {
        output @ _ => match output.status.success() {
            true => {}
            false => {
                eprintln!(
                    "clang output: {}",
                    String::from_utf8(output.stdout).unwrap()
                );
                panic!("compile failed")
            }
        },
    }
    Ok(PathBuf::from(format!(
        "{}/{}",
        build_dir.display(),
        target_lib
    )))
}

mod test {

    #[test]
    fn test_compile() {
        use super::*;
        let src = r#"extern AKR_JIT_DLL_EXPORT int add(int x, int y){return x+y;}
        extern AKR_JIT_DLL_EXPORT int mul(int x, int y){return x*y;}"#;
        let path = compile(src, "add").unwrap();
        unsafe {
            let lib = libloading::Library::new(dbg!(path)).unwrap();
            let add: libloading::Symbol<unsafe extern "C" fn(i32, i32) -> i32> =
                lib.get(b"add\0").unwrap();
            let mul: libloading::Symbol<unsafe extern "C" fn(i32, i32) -> i32> =
                lib.get(b"mul\0").unwrap();
            assert_eq!(add(1, 2), 3);
            assert_eq!(mul(2, 4), 8);
        }
    }
}
