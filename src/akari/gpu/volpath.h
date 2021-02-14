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
#include <akari/util.h>
#include <akari/render_xpu.h>
#include <akari/gpu/kernel.h>
#include <akari/gpu/soa.h>
namespace akari::gpu::kernel {
    using namespace akari::render;
    struct PathState {
        enum KernelState {
            KERNEL_RAYGEN,
            
        };
        KernelState state;
        ivec2 pixel;
        Ray ray;
        Sampler<GPU> sampler;
    };

    struct KernelGlobals {
        const Camera<GPU> *camera = nullptr;
        SOA<PathState> path_states;
    };
    struct VolPathKernels {
        Kernel advance;
    };

    VolPathKernels load_kernels();
} // namespace akari::gpu