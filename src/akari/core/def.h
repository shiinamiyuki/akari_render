// MIT License
//
// Copyright (c) 2020 椎名深雪
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// Predefined macros:
/*
AKR_ENABLE_GPU
AKR_ENABLE_CPU (always on)
AKR_ENABLE_EMBREE
AKR_GPU_BACKEND_CUDA
AKR_GPU_BACKEND_SYCL
AKR_GPU_BACKEND_METAL
AKR_PLATFORM_WINDOWS
AKR_PLATFORM_LINUX
*/

#define AKR_CPU
#define AKR_GPU

#if defined(__CUDA_ARCH__)
#    define AKR_GPU_CODE
#endif

#ifdef AKR_ENABLE_GPU
#    ifdef AKR_GPU_BACKEND_CUDA
#        include <cuda_runtime.h>
#        undef AKR_CPU
#        undef AKR_GPU
#        define AKR_CPU __host__
#        define AKR_GPU __device__
#    endif
#endif
#define AKR_XPU AKR_GPU AKR_CPU