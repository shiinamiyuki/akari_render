<h1 align="center">AkariRender</h1>
<h5 align="center">High Performance CPU<del>/GPU</del> Physically Based Renderer</h5>
<h5 align="center"> ⚠ The new based on <a href=https://github.com/LuisaGroup/luisa-compute-rs>LuisaCompute</a> is updated! ⚠ </h5>

<!-- ![](gallery/beauty4k.png) -->
![](gallery/psor.png)

AkariRender is a CPU/GPU physically based renderer written in Rust and powered by *LuisaCompute*.



## Features
- RGB/Spectral Rendering (WIP)
- Path Tracing
- Path Tracing with Metropolis Sampling
- Gradient Domain Path Tracing (WIP)


## Build Requirements
If you are using < Windows 10, please upgrade to Windows 10 or above.
- CMake > 3.23
- clang++ in PATH

## Build & Run
```
cargo build --relase
cargo run --bin akari_cli -- -d cpu -s scenes/cbox/scene.json -m scenes/cbox/test.json
```
