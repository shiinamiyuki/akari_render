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

#include <akari/kernel/cuda/util.h>
#include <akari/core/parallel.h>
#include <akari/core/options.h>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

namespace akari::gpu {
    std::pair<cudaEvent_t, cudaEvent_t> get_profiler_events(const char *description);
    void print_kernel_stats();
    template <typename F>
    inline int get_block_size(const char *description, F kernel) {
        static std::unordered_map<std::type_index, int> kernelBlockSizes;

        std::type_index index = std::type_index(typeid(F));

        auto iter = kernelBlockSizes.find(index);
        if (iter != kernelBlockSizes.end())
            return iter->second;

        int minGridSize, blockSize;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
        kernelBlockSizes[index] = blockSize;

        return blockSize;
    }

    template <typename KernelName = void,typename F>
    __global__ void _kernel_wrapper(F func, int nItems) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= nItems)
            return;
        func(tid);
    }
#ifdef AKR_PLATFORM_WINDOWS
    #define AKR_GPU_LAMBDA(...) [=,*this] AKR_GPU(__VA_ARGS__) mutable
#else
    #define AKR_GPU_LAMBDA(...) [=] AKR_GPU(__VA_ARGS__)
#endif
 #define AKR_CPU_LAMBDA(...) [=,*this](__VA_ARGS__) mutable
    template <class KernelName = void, typename F>
    void launch(const char *name, int nItems, F func) {
        auto kernel = &_kernel_wrapper<KernelName, F>;
        int blockSize = get_block_size(name, kernel);
        int gridSize = (nItems + blockSize - 1) / blockSize;
        AKR_ASSERT(blockSize >= 32);
        auto event = get_profiler_events(name);
        nvtxRangePush(name);
        cudaEventRecord(event.first);       
        kernel<<<gridSize, blockSize>>>(func, nItems);
        cudaEventRecord(event.second);
        nvtxRangePop();

        //  CUDA_CHECK(cudaDeviceSynchronize());
    }
    template<typename F>
    void launch_single(const char * name, F func){
        launch(name, 1, [=]AKR_GPU (int )mutable{func();});
    }
    template <typename F>
    void launch_cpu(const char *name, int nItems, F func) {
        parallel_for(nItems, [&](uint32_t tid, auto _){
            func(tid);
        });
    }
} // namespace akari::gpu
