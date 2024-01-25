<h1 align="center">AkariRender</h1>
<h5 align="center">High Performance CPU/GPU Physically Based Renderer</h5>
<h5 align="center"> ⚠ The new version based on <a href=https://github.com/LuisaGroup/luisa-compute-rs>LuisaCompute</a> is updated! ⚠ </h5>

<!-- ![](gallery/beauty4k.png) -->
![](gallery/psor.png)

AkariRender is a CPU/GPU physically based renderer written in Rust and powered by *LuisaCompute*.


## Features
- Loading Blender scene
- Supports a subset of Cycles shader graph via JIT
- RGB/Spectral Rendering (WIP)
- Path Tracing
- Path Tracing with Metropolis Sampling
- Gradient Domain Path Tracing


## Building:
If you are using < Windows 10, please upgrade to Windows 10 or above.
- CMake > 3.23
- Ninja
- Clone Blender 4.0 source code from `blender-v4.0-release` branch
- Put path to blender source in `blender_src_path.txt`
- Clone [LuisaCompute](https://github.com/LuisaGroup/luisa-compute-rs) alongside this repo
  
If you intend to run the renderer on cpu, the following runtime requirement must be satisfied:
- clang++ in `PATH`
- llvm dynamic library of the same version. For Windows users, it is the `LLVM-C.dll`.

## Run
```
cargo run --release --bin akari-cli -- -d (cpu|cuda|dx|metal) -s scenes/cbox/scene.json -m scenes/cbox/test.json
```
