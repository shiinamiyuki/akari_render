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

#include "accel.h"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <akari/gpu/cuda/check.h>

static void logCallback(unsigned int level, const char *tag, const char *message, void *cbdata) {
    if (level <= 2)
        spdlog::info("OptiX: {}: {}", tag, message);
    else
        spdlog::info("OptiX: {}: {}", tag, message);
}

namespace akari::gpu {
    OptixAccel::OptixAccel(std::shared_ptr<Device> device) : device(device) {
        CUcontext cudaContext;
        CU_CHECK(cuCtxGetCurrent(&cudaContext));
        AKR_ASSERT(cudaContext != nullptr);
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions ctxOptions = {};
#ifndef NDEBUG
        ctxOptions.logCallbackLevel = 4; // status/progress
#else
        ctxOptions.logCallbackLevel = 2; // error
#endif
        ctxOptions.logCallbackFunction = logCallback;
#if (OPTIX_VERSION >= 70200)
        ctxOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &ctxOptions, &optix_context));
    }
    void OptixAccel::build(scene::P<scene::SceneGraph> graph) {
        auto dispatcher = device->new_dispatcher();
        using scene::P;
        scene = Scene();
        std::unordered_map<P<scene::Mesh>, uint32_t> mesh_id_map;
        auto create_mesh = [&](P<scene::Mesh> mesh_node) -> uint32_t {
            if (mesh_id_map.find(mesh_node) != mesh_id_map.end()) {
                return mesh_id_map.at(mesh_node);
            }
            Buffer<float> vertices  = device->allocate_buffer<float>(3 * mesh_node->vertices.size());
            Buffer<float> normals   = device->allocate_buffer<float>(3 * mesh_node->normals.size());
            Buffer<float> texcoords = device->allocate_buffer<float>(2 * mesh_node->texcoords.size());
            Buffer<int> indices     = device->allocate_buffer<int>(3 * mesh_node->indices.size());
            vertices.upload(dispatcher, 0, vertices.size(), (const float *)mesh_node->vertices.data());
            normals.upload(dispatcher, 0, normals.size(), (const float *)mesh_node->normals.data());
            texcoords.upload(dispatcher, 0, texcoords.size(), (const float *)mesh_node->texcoords.data());
            indices.upload(dispatcher, 0, indices.size(), (const int *)mesh_node->indices.data());
            auto mesh = std::make_shared<Mesh>(std::move(vertices), std::move(normals), std::move(texcoords),
                                               std::move(indices));
            auto id   = scene.meshes.size();
            mesh_id_map[mesh_node] = id;
            scene.meshes.push_back(mesh);
            return id;
        };
        std::unordered_map<P<scene::Instance>, uint32_t> instance_id_map;
        auto create_instance = [&](P<scene::Instance> instance_node) -> MeshInstance {
            if (instance_id_map.find(instance_node) != instance_id_map.end()) {
                return scene.instances[instance_id_map.at(instance_node)];
            }
            auto mesh_node = instance_node->mesh;
            auto id        = create_mesh(mesh_node);
            MeshInstance instance;
            instance.mesh_id               = id;
            auto id                        = scene.instances.size();
            instance_id_map[instance_node] = id;
            scene.instances.push_back(instance);
            return instance;
        };
    }

} // namespace akari::gpu