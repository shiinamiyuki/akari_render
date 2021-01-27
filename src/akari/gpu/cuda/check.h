// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <spdlog/spdlog.h>
#define OPTIX_CHECK(EXPR)                                                                                              \
    [&] {                                                                                                              \
        OptixResult res = EXPR;                                                                                        \
        if (res != OPTIX_SUCCESS)                                                                                      \
            spdlog::error("OptiX call " #EXPR " failed with code {}: \"{}\"", int(res), optixGetErrorString(res));     \
    }()
#define CUDA_CHECK(EXPR)                                                                                               \
    [&] {                                                                                                              \
        if (EXPR != cudaSuccess) {                                                                                     \
            cudaError_t error = cudaGetLastError();                                                                    \
            spdlog::error("CUDA error: {}", cudaGetErrorString(error));                                                \
        }                                                                                                              \
    }()

#define CU_CHECK(EXPR)                                                                                                 \
    [&] {                                                                                                              \
        CUresult result = EXPR;                                                                                        \
        if (result != CUDA_SUCCESS) {                                                                                  \
            const char *str;                                                                                           \
            AKR_ASSERT(CUDA_SUCCESS == cuGetErrorString(result, &str));                                                  \
            spdlog::error("CUDA error: {}", str);                                                                      \
        }                                                                                                              \
    }()