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
#include <akari/gpu/cuda/impl.h>
#include <akari/gpu/cuda/ptx/optix_ptx.h>

static void logCallback(unsigned int level, const char *tag, const char *message, void *cbdata) {
    if (level <= 2)
        spdlog::info("OptiX: {}: {}", tag, message);
    else
        spdlog::info("OptiX: {}: {}", tag, message);
}

namespace akari::gpu {
    using namespace accel;
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    OptixAccel::OptixAccel(std::shared_ptr<Device> device) : device(device), dispatcher(device->new_dispatcher()) {
        CUDA_CHECK(cudaFree(nullptr));
        stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
        CUcontext cudaContext;
        CU_CHECK(cuCtxGetCurrent(&cudaContext));
        AKR_ASSERT(cudaContext != nullptr);
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions ctxOptions = {};
#ifndef NDEBUG
        ctxOptions.logCallbackLevel = 4; // status/progress
#else
        ctxOptions.logCallbackLevel     = 2; // error
#endif
        ctxOptions.logCallbackFunction = logCallback;
#if (OPTIX_VERSION >= 70200)
        ctxOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &ctxOptions, &optix_context));
        create_module();
    }
    void OptixAccel::create_module() {
        // OptiX module
        OptixModuleCompileOptions moduleCompileOptions = {};
        // TODO: REVIEW THIS
        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
        moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags       = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineCompileOptions.usesMotionBlur              = false;
        pipelineCompileOptions.numPayloadValues            = 3;
        pipelineCompileOptions.numAttributeValues          = 4;
        // OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.exceptionFlags =
            (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_DEBUG);
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth            = 2;
#ifndef NDEBUG
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
        pipelineLinkOptions.debugLevel  = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_WITH_LOG(optixModuleCreateFromPTX(optix_context, &moduleCompileOptions, &pipelineCompileOptions,
                                                      (const char *)optix_ptx, optix_ptx_size, log, &logSize,
                                                      &optix_module),
                             log);

        spdlog::info("Optix: {}", log);
    }
    OptixBuildInput OptixAccel::get_mesh_build_input(const Mesh &mesh) {
        OptixBuildInput input = {};

        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        vertx_buf_ptrs.push_back(0);
        CUdeviceptr &vertx_buf_ptr = vertx_buf_ptrs.back();
        {
            auto *impl                              = dynamic_cast<CUDABuffer *>(mesh.vertices.impl_mut());
            input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
            input.triangleArray.numVertices         = mesh.vertices.size() / 3;
            vertx_buf_ptr                           = (CUdeviceptr)impl->device_ptr();
            input.triangleArray.vertexBuffers       = &vertx_buf_ptr;
        }
        {
            auto *impl                             = dynamic_cast<CUDABuffer *>(mesh.indices.impl_mut());
            input.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
            input.triangleArray.numIndexTriplets   = mesh.indices.size() / 3;
            input.triangleArray.indexBuffer        = (CUdeviceptr)impl->device_ptr();
        }

        // triangleInputFlags[buildIndex] = getOptixGeometryFlags(true, alphaTextureHandle, materialHandle);
        input.triangleArray.flags = &geom_flags;

        input.triangleArray.numSbtRecords               = 1;
        input.triangleArray.sbtIndexOffsetBuffer        = reinterpret_cast<CUdeviceptr>(nullptr);
        input.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
        input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
        // OptixAccelBuildOptions accelOptions             = {};
        // accelOptions.buildFlags            = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
        // OPTIX_BUILD_FLAG_PREFER_FAST_TRACE); accelOptions.motionOptions.numKeys = 1; accelOptions.operation =
        // OPTIX_BUILD_OPERATION_BUILD; OPTIX_CHECK(optixAccelBuild(optix_context, stream,&accelOptions,&input,))
        return input;
    }
    OptixTraversableHandle OptixAccel::build_bvh(const std::vector<OptixBuildInput> &build_inputs) {
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags             = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        accelOptions.motionOptions.numKeys  = 1;
        accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes blasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context, &accelOptions, build_inputs.data(), build_inputs.size(),
                                                 &blasBufferSizes));

        // void *compactedSizeBufferPtr;
        // CUDA_CHECK(cudaMallocManaged(&compactedSizeBufferPtr, sizeof(uint64_t)));
        auto compactedSizeBuffer = device->allocate_buffer<uint64_t>(1);
        uint64_t compactedSize   = 0;
        OptixAccelEmitDesc emitDesc;
        emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = (CUdeviceptr) dynamic_cast<CUDABuffer *>(compactedSizeBuffer.impl_mut())->device_ptr();

        AKR_ASSERT(blasBufferSizes.tempSizeInBytes > 0);
        AKR_ASSERT(blasBufferSizes.outputSizeInBytes > 0);
        // Allocate buffers.
        void *tempBuffer;
        CUDA_CHECK(cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes));
        AKR_CHECK(tempBuffer != nullptr);
        void *outputBuffer;
        CUDA_CHECK(cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes));
        AKR_CHECK(outputBuffer != nullptr);
        OptixTraversableHandle traversableHandle = 0;
        CUDA_CHECK(cudaDeviceSynchronize());
        OPTIX_CHECK(optixAccelBuild(optix_context, stream, &accelOptions, build_inputs.data(), build_inputs.size(),
                                    CUdeviceptr(tempBuffer), blasBufferSizes.tempSizeInBytes, CUdeviceptr(outputBuffer),
                                    blasBufferSizes.outputSizeInBytes, &traversableHandle, &emitDesc, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
        compactedSizeBuffer.download(dispatcher, 0, 1, &compactedSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        gpu_bvh_bytes += compactedSize;
        void *asBuffer;
        CUDA_CHECK(cudaMalloc(&asBuffer, compactedSize));

        OPTIX_CHECK(optixAccelCompact(optix_context, stream, traversableHandle, CUdeviceptr(asBuffer), compactedSize,
                                      &traversableHandle));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(tempBuffer));
        CUDA_CHECK(cudaFree(outputBuffer));

        return traversableHandle;
    }
    void OptixAccel::build(scene::P<scene::SceneGraph> graph) {
        using scene::P;
        scene = Scene();
        std::unordered_map<P<scene::Mesh>, uint32_t> mesh_id_map;
        auto create_mesh = [&](P<scene::Mesh> mesh_node) -> uint32_t {
            if (mesh_id_map.find(mesh_node) != mesh_id_map.end()) {
                return mesh_id_map.at(mesh_node);
            }
            Buffer<float> vertices   = device->allocate_buffer<float>(3 * mesh_node->vertices.size());
            Buffer<float> normals    = device->allocate_buffer<float>(3 * mesh_node->normals.size());
            Buffer<float> texcoords  = device->allocate_buffer<float>(2 * mesh_node->texcoords.size());
            Buffer<uint32_t> indices = device->allocate_buffer<uint32_t>(3 * mesh_node->indices.size());
            vertices.upload(dispatcher, 0, vertices.size(), (const float *)mesh_node->vertices.data());
            normals.upload(dispatcher, 0, normals.size(), (const float *)mesh_node->normals.data());
            texcoords.upload(dispatcher, 0, texcoords.size(), (const float *)mesh_node->texcoords.data());
            indices.upload(dispatcher, 0, indices.size(), (const uint32_t *)mesh_node->indices.data());
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
            auto mesh_id   = create_mesh(mesh_node);
            MeshInstance instance;
            instance.mesh_id               = mesh_id;
            auto inst_id                   = scene.instances.size();
            instance_id_map[instance_node] = inst_id;
            scene.instances.push_back(instance);
            return instance;
        };
        for (auto &mesh_node : graph->meshes) {
            (void)create_mesh(mesh_node);
        }
        for (auto &instance_node : graph->instances) {
            (void)create_instance(instance_node);
        }
        dispatcher.wait();
        CUDA_CHECK(cudaDeviceSynchronize());
        for (auto &mesh : scene.meshes) {
            std::vector<OptixBuildInput> inputs;
            inputs.push_back(get_mesh_build_input(*mesh));
            mesh_handles.push_back(build_bvh(inputs));
        }
        std::vector<OptixInstance> ias_instances;
        for (auto &instance : scene.instances) {
            OptixInstance gasInstance = {};
            float identity[12]        = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
            memcpy(gasInstance.transform, identity, 12 * sizeof(float));
            gasInstance.visibilityMask    = 255;
            gasInstance.traversableHandle = mesh_handles.at(instance.mesh_id);
            gasInstance.sbtOffset         = 0;
            ias_instances.push_back(gasInstance);
        }
        {
            ias_instances_buf.emplace(device->allocate_buffer<OptixInstance>(ias_instances.size()));
            ias_instances_buf->upload(dispatcher, 0, ias_instances.size(), ias_instances.data());
            auto *impl = dynamic_cast<CUDABuffer *>(ias_instances_buf->impl_mut());
            // Build the top-level IAS
            OptixBuildInput buildInput               = {};
            buildInput.type                          = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            buildInput.instanceArray.instances       = CUdeviceptr(impl->device_ptr());
            buildInput.instanceArray.numInstances    = ias_instances.size();
            std::vector<OptixBuildInput> buildInputs = {buildInput};
            root_traversable                         = build_bvh({buildInput});
        }
    }

} // namespace akari::gpu