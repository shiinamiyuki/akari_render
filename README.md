<h1 align="center">AkariRender</h1>
<h5 align="center">High Performance CPU/GPU Physically Based Renderer (and More!)</h5>

![](gallery/beauty4k.png)

AkariRender is a highly modular CPU/GPU physically based renderer written in C++17.

### Status
[![Build Status](https://travis-ci.org/shiinamiyuki/AkariRender.svg?branch=master)](https://travis-ci.org/shiinamiyuki/AkariRender)

## Features
 - <del> Optional GPU rendering using CUDA + Optix7(WIP)</del> 
 - Optional Embree backend
 - Optional OpenImageDenoise
 - Optional Optix AI Denoiser
 - Optional Network Rendering

## Usage
```bash
source setpath.sh
mkdir workspace && cd workspace
# import existing mesh; generates a mesh.akari file
akari-import mesh.obj mesh.akari
# create the top level scene file
vim scene.akari
# render !
akari scene.akari --spp 64
# render with denoiser with 4x super sampling
akari scene.akari --spp 64 --denoise --ss 4
```

## Build from Source
Required packages: 
  - zlib if AKR_ENABLE_OPENEXR (I cannot get in source build to work)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
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
