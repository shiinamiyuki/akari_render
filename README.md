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
AkariRender uses a custom package manager called [useless](https://github.com/shiinamiyuki/useless). It's completely useless but still more useful than vcpkg/system package manager.


```bash
git clone --recursive https://github.com/shiinamiyuki/AkariRender
cd useless
python main.py install embree assimp cereal glm cxxopts pybind11 openexr spdlog stb
cd ..
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 8
```



