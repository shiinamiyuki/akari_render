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

#include <akari/core/math.h>
#include <akari/render/cuda/util.h>
#include <akari/render/scene.h>
#include <optix.h>
#include <optix_host.h>

namespace akari::gpu {
    using render::MeshInstance;
    using render::Scene;
    class AKR_EXPORT GPUAccel {
        struct OptixState {
            OptixDeviceContext context;
            OptixTraversableHandle gas_handle;
            OptixModule ptx_module = 0;
            OptixPipelineCompileOptions pipeline_compile_options = {};
            OptixPipeline pipeline = 0;
            OptixProgramGroup raygen_group;
            OptixProgramGroup radiance_closest_hit_group;
            OptixProgramGroup radiance_miss_group;
            OptixProgramGroup shadow_miss_group;
            OptixProgramGroup shadow_any_hit_group;
            OptixShaderBindingTable mega_kernel_sbt = {};
        };
        OptixState state;
        Allocator<> allocator;
        void init_optix();
        OptixBuildInput build(const MeshInstance &instance);
        void build(const std::vector<OptixBuildInput> &inputs);
        void build_ptx_module();
        void build_pipeline();
        void build_sbt();

      public:
        GPUAccel(Allocator<> allocator) : allocator(allocator) { init_optix(); }
        void build(const Scene *scene);
    };
} // namespace akari::gpu