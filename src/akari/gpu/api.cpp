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

#include <akari/gpu/api.h>
#include <akari/gpu/cuda/accel.h>
#include <akari/gpu/cuda/impl.h>
#include <akari/gpu/cuda/impl.h>
#include <akari/gpu/volpath.h>
#include <spdlog/spdlog.h>
namespace akari::gpu {
    void render_scenegraph(scene::P<scene::SceneGraph> graph, const std::string &backend) {
        graph->commit();
#ifdef AKR_BACKEND_CUDA
        if (backend == "cuda") {
            auto device     = create_cuda_device();
            auto dispatcher = device->new_dispatcher();
            OptixAccel accel(device);
            accel.build(graph);
            auto kernels = kernel::load_kernels();
            kernels.advance.launch(dispatcher, uvec3(1024, 1, 1), uvec3(64, 1, 1), {});
            dispatcher.wait();
            return;
        }
#endif
        spdlog::error("unknown backend: {}", backend);
    }
} // namespace akari::gpu