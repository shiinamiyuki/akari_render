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

#include <akari/core/logger.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>

namespace akari {
#define CUDA_CHECK(EXPR)                                                                                               \
    do {                                                                                                               \
        if (EXPR != cudaSuccess) {                                                                                     \
            cudaError_t error = cudaGetLastError();                                                                    \
            fatal("CUDA error: {} ", cudaGetErrorString(error));                                                       \
            fatal("Calling {} at {}:{}", #EXPR, __FILE__, __LINE__);                                                   \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#define OPTIX_CHECK(EXPR)                                                                                              \
    ([&] {                                                                                                             \
        auto res = (EXPR);                                                                                             \
        if (res != OPTIX_SUCCESS) {                                                                                    \
            fatal("Optix error: {} ", optixGetErrorString(res));                                                       \
            fatal("Calling {} at {}:{}", #EXPR, __FILE__, __LINE__);                                                   \
            std::abort();                                                                                              \
        }                                                                                                              \
    })()
} // namespace akari