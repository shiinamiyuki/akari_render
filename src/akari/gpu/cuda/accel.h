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
#include <list>
#include <optix.h>
#include <optix_host.h>
#include <driver_types.h>

#include <akari/scenegraph.h>
#include <akari/gpu/device.h>
// #include <akari/gpu/scene.h>
namespace akari::gpu {
    namespace accel {
        class Mesh {
          public:
            Buffer<float> vertices;
            Buffer<float> normals;
            Buffer<float> texcoords;
            Buffer<uint32_t> indices;
            Mesh(Buffer<float> vertices, Buffer<float> normals, Buffer<float> texcoords, Buffer<uint32_t> indices)
                : vertices(std::move(vertices)), normals(std::move(normals)), texcoords(std::move(texcoords)),
                  indices(std::move(indices)) {}
        };
        class MeshInstance {
          public:
            Transform transform;
            uint32_t mesh_id     = uint32_t(-1);
            uint32_t material_id = uint32_t(-1);
            uint32_t volume_id   = uint32_t(-1);
        };

        // struct Material {

        // };
        class Scene {
          public:
            std::vector<std::shared_ptr<Mesh>> meshes;
            std::vector<MeshInstance> instances;

            // std::vector<
        };
    } // namespace accel
    class OptixAccel {
        std::shared_ptr<Device> device;
        Dispatcher dispatcher;
        cudaStream_t stream;
        OptixDeviceContext optix_context;
        OptixModule optix_module;
        OptixTraversableHandle root_traversable;
        OptixDeviceContext context;
        accel::Scene scene;
        uint32_t geom_flags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        OptixBuildInput get_mesh_build_input(const accel::Mesh &mesh);
        OptixTraversableHandle build_bvh(const std::vector<OptixBuildInput> &build_inputs);
        std::vector<OptixTraversableHandle> mesh_handles;
        size_t gpu_bvh_bytes = 0;
        astd::optional<Buffer<OptixInstance>> ias_instances_buf;
        std::list<CUdeviceptr> vertx_buf_ptrs;
        void create_module();

      public:
        OptixAccel(std::shared_ptr<Device> device);
        void build(scene::P<scene::SceneGraph> graph);
    };
} // namespace akari::gpu