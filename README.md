<h1 align="center">AkariRender</h1>
<h5 align="center">High Performance CPU/GPU Physically Based Renderer (and More!)</h5>

![](gallery/beauty4k.png)

AkariRender is a CPU/GPU physically based renderer written in C++17.

### Status
[![Build Status](https://travis-ci.org/shiinamiyuki/AkariRender.svg?branch=master)](https://travis-ci.org/shiinamiyuki/AkariRender)

## Features
 - <del> Optional GPU rendering using CUDA + Optix7(WIP)</del> 
 - Optional Embree backend
 - Optional denoiser using OpenImageDenoise

## Build from Source
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# when using CUDA + OptiX 7
cmake .. -DCMAKE_BUILD_TYPE=Release -DAKR_OPTIX_PATH=[path to optix7]
```

### To use Embree
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DAKR_ENABLE_EMBREE=ON
```

### To use OIDN
```bash
cd external
git clone --recursive https://github.com/OpenImageDenoise/oidn
## install prerequisites (libtbb-dev, ispc) as required by oidn 
cd ..
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DAKR_ENABLE_OIDN=ON
```
