<h1 align="center">AkariRender</h1>
<h5 align="center">High Performance CPU/GPU Physically Based Renderer (and More!)</h5>

![](gallery/beauty4k.png)

AkariRender is a highly modular CPU/GPU physically based renderer written in C++17.

### Status
[![Build Status](https://travis-ci.org/shiinamiyuki/AkariRender.svg?branch=master)](https://travis-ci.org/shiinamiyuki/AkariRender)

## Features
- Modified Instant Radiosity (supports all kinds of BSDFs)
- Unidirectional Path Tracing
- Practical Path Guiding with improvments
- Stratified MCMC
- [WIP] Specular Manifold Sampling
- [WIP] Python binding
## TODO
- Realtime Rendering

## Build
### If you don't have / don't know vcpkg
```bash
git clone --recursive https://github.com/shiinamiyuki/AkariRender
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release # vcpkg will download and install dependencies
make -j 8
```

### If you have vcpkg installed
Pass CMAKE_TOOLCHAIN_FILE to cmake to force using your own vcpkg

Dependencies:
- OpenImageIO
- Embree3
- glm
- cereal
- assimp
- pybind11
- cxxopts


