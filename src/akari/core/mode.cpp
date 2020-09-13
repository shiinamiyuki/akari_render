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

#include <akari/core/mode.h>
#ifdef AKR_ENABLE_GPU
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <akari/kernel/cuda/util.h>
#endif
namespace akari {
#ifdef AKR_ENABLE_GPU
    class cuda_unified_memory_resource : public astd::pmr::memory_resource {
      public:
        void *do_allocate(size_t bytes, size_t alignment) {
            void *p;
            CUDA_CHECK(cudaMallocManaged(&p, bytes));
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) { CUDA_CHECK(cudaFree(p)); }
        bool do_is_equal(const memory_resource &other) const noexcept { return &other == this; }
    };
    static cuda_unified_memory_resource _cuda_unified_memory_resource;
#endif
    namespace _mode_internal {
        ComputeDevice cur_device = ComputeDevice::cpu;
    }
#ifndef AKR_ENABLE_GPU
    void set_device_gpu() {
        fatal("gpu rendering is not supported\n");
        std::abort();
    }
#else
    void set_device_gpu() { _mode_internal::cur_device = ComputeDevice::gpu; }
#endif
    void set_device_cpu() { _mode_internal::cur_device = ComputeDevice::cpu; }
    AKR_EXPORT ComputeDevice get_device() { return _mode_internal::cur_device; }
    astd::pmr::memory_resource *get_device_memory_resource() {
        if (get_device() == ComputeDevice::cpu) {
            return astd::pmr::get_default_resource();
        } else {
            return &_cuda_unified_memory_resource;
        }
    }
} // namespace akari