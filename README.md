<h1 align="center">AkariRender</h1>
<h5 align="center">Extensible Physically Based Renderer (and More!)</h5>

![](gallery/final-bdpt.png)

AkariRender is a CPU/GPU physically based renderer written in C++17.
### Status
[![Build Status](https://travis-ci.org/shiinamiyuki/AkariRender.svg?branch=master)](https://travis-ci.org/shiinamiyuki/AkariRender)

## Features
 - Optional Embree backend
 - Optional Intel OpenImageDenoise

## Build from Source
```bash
cp ./resources/config.default.py akari.conf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```


