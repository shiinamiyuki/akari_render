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

#include <akari/gpu/cuda/accel.h>
#include <akari/gpu/volpath.h>
#include <akari/gpu/cuda/impl.h>
#include <akari/gpu/cuda/ptx/volpath_ptx.h>
namespace akari::gpu {
    VolPathKernels load_kernels() {
        spdlog::info("loading kernels");
        CUmodule module;
        CU_CHECK(cuModuleLoadData(&module, volpath_ptx));
        CUfunction func;
        CU_CHECK(cuModuleGetFunction(&func, module, "volpath_advance"));
        return VolPathKernels{Kernel(std::make_unique<CUDAKernel>(func))};
    }
} // namespace akari::gpu