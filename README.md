<h1 align="center">AkariRender v3 (Experimental)</h1>
<h5 align="center">High Performance CPU/GPU Physically Based Renderer</h5>

![](gallery/classroom.png)
Scene by Christophe Seux (CC0)

AkariRender is a CPU/GPU physically based renderer written in Rust and powered by *LuisaCompute*.

## Build and Run
To build and run the project, you need to have the following dependencies installed:
- [CMake](https://cmake.org/)
- [Ninja](https://ninja-build.org/)
- C++23 compiler (MSVC, GCC, Clang)
- [Rust](https://www.rust-lang.org/)
- [Python 3.11+](https://www.python.org/)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (Optional, for GPU rendering)
<!-- - [CUDA 12.2](https://developer.nvidia.com/cuda) (Optional, for GPU rendering) -->

Building and running the project is handled by the `run.py` script, which has similar interface to `cargo`. For example, to build the project, simply run:
```bash
python run.py build # a debug build
python run.py build --release # a release build

```
