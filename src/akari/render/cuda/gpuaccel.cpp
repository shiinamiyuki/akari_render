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

#include <akari/render/cuda/gpuaccel.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
extern "C" {
extern unsigned char AKR_EMBEDDED_PTX[];
}
namespace akari::gpu {
    using namespace render;
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    void GPUAccel::init_optix() {
        CUDA_CHECK(cudaFree(0));

        OptixDeviceContext context;
        CUcontext cu_ctx = 0; // zero means take the current context
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = [](unsigned int level, const char *tag, const char *message, void *cbdata) {
            if (level == 1) {
                fatal("optix: [{}] {}", tag, message);
            } else if (level == 2) {
                error("optix: [{}] {}", tag, message);
            } else if (level == 3) {
                warning("optix: [{}] {}", tag, message);
            } else if (level == 4) {
                info("optix: [{}] {}", tag, message);
            }
        };
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

        state.context = context;
    }
    void GPUAccel::build(const std::vector<OptixBuildInput> &inputs) {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, inputs.data(),
                                                 inputs.size(), // num_build_inputs
                                                 &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

        // non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t compactedSizeOffset = (gas_buffer_sizes.outputSizeInBytes + 7ull) & ~7ull;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
                              compactedSizeOffset + 8));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

        OPTIX_CHECK(optixAccelBuild(state.context,
                                    0, // CUDA stream
                                    &accel_options, inputs.data(),
                                    inputs.size(), // num build inputs
                                    d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
                                    d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes,
                                    &state.gas_handle,
                                    &emitProperty, // emitted property list
                                    1              // num emitted properties
                                    ));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
    }
    OptixBuildInput GPUAccel::build(const MeshInstance &instance) {
        //
        // Build triangle GAS
        //
        uint32_t triangle_input_flags[1] = // One per SBT record for this build input
            {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
        triangle_input.triangleArray.numVertices = static_cast<uint32_t>(instance.vertices.size() / 3);
        triangle_input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr const *>(&instance.vertices.data());
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;
        triangle_input.triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(nullptr);
        triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
        return triangle_input;
    }
    void GPUAccel::build_pipeline() {
        OptixProgramGroupOptions program_group_options = {};

        char log[2048];
        size_t sizeof_log = sizeof(log);

        {
            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = state.ptx_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

            OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &raygen_prog_group_desc,
                                                    1, // num program groups
                                                    &program_group_options, log, &sizeof_log, &state.raygen_group));
        }

        {
            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = state.ptx_module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &miss_prog_group_desc,
                                                    1, // num program groups
                                                    &program_group_options, log, &sizeof_log,
                                                    &state.radiance_miss_group));
            memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = state.ptx_module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &miss_prog_group_desc,
                                                    1, // num program groups
                                                    &program_group_options, log, &sizeof_log,
                                                    &state.shadow_miss_group));
        }

        {
            OptixProgramGroupDesc hit_prog_group_desc = {};
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &hit_prog_group_desc,
                                                    1, // num program groups
                                                    &program_group_options, log, &sizeof_log,
                                                    &state.radiance_closest_hit_group));

            memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__anyhit__shadow";
            sizeof_log = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(state.context, &hit_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &state.shadow_any_hit_group));
        }

        OptixProgramGroup program_groups[] = {state.raygen_group, state.radiance_closest_hit_group,
                                              state.shadow_any_hit_group, state.radiance_miss_group,
                                              state.shadow_miss_group};
        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = 2;
#ifndef NDEBUG
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
        OPTIX_CHECK_LOG(optixPipelineCreate(state.context, &state.pipeline_compile_options, &pipeline_link_options,
                                            program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log,
                                            &sizeof_log, &state.pipeline));
    }
    void GPUAccel::build_sbt() {
        {
            RaygenRecord *raygen_record = allocator.new_object<RaygenRecord>();
            OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_closest_hit_group, raygen_record));
            state.mega_kernel_sbt.raygenRecord = (CUdeviceptr)raygen_record;
        }
        {
            MissRecord *miss_record = allocator.new_object<MissRecord>();
            OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_closest_hit_group, miss_record));
            state.mega_kernel_sbt.missRecordBase = (CUdeviceptr)miss_record;
            state.mega_kernel_sbt.missRecordStrideInBytes = sizeof(MissRecord);
            state.mega_kernel_sbt.missRecordCount = 2;
        }
    }
    void GPUAccel::build(const Scene *scene) {
        std::vector<OptixBuildInput> inputs;
        for (auto &instance : scene->meshes) {
            inputs.emplace_back(build(instance));
        }
        build(inputs);
        build_ptx_module();
    }
    void GPUAccel::build_ptx_module() {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        state.pipeline_compile_options.usesMotionBlur = false;
        state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        state.pipeline_compile_options.numPayloadValues = 2;
        state.pipeline_compile_options.numAttributeValues = 2;

#ifndef NDEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should
               // only be done during development.
        state.pipeline_compile_options.exceptionFlags =
            OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
        state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        const std::string_view ptx((const char *)AKR_EMBEDDED_PTX);

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(state.context, &module_compile_options,
                                                 &state.pipeline_compile_options, ptx.data(), ptx.size(), log,
                                                 &sizeof_log, &state.ptx_module));
    }
} // namespace akari::gpu