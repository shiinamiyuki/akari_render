<h1 align="center">AkariRender</h1>
<h5 align="center">High Performance CPU/GPU Physically Based Renderer</h5>
<h5 align="center"> ⚠ The new based on <a href=https://github.com/LuisaGroup/luisa-compute-rs>LuisaCompute</a> is updated! ⚠ </h5>

<!-- ![](gallery/beauty4k.png) -->
![](gallery/psor.png)

AkariRender is a CPU/GPU physically based renderer written in Rust and powered by *LuisaCompute*.


## Features
- Loading Blender scene
- Cycles shader graph
- RGB/Spectral Rendering (WIP)
- Path Tracing
- Path Tracing with Metropolis Sampling
- Gradient Domain Path Tracing (WIP)


## Building:
If you are using < Windows 10, please upgrade to Windows 10 or above.
- CMake > 3.23
- Ninja
- Blender 4.0 source code
If you intend to run the renderer on cpu, the following runtime requirement must be satisified
- clang++ in PATH
- llvm dynamic library of the same version. For Windows users, it is the `LLVM-C.dll`.

## Build & Run
Put the path to Blender 4.0 source in `akari_scenegraph/blender_src_path.txt`
```
cargo build --relase
cargo run --release --bin akari_cli -- -d (cpu|cuda|dx|metal) -s scenes/cbox/scene.json -m scenes/cbox/test.json
```
