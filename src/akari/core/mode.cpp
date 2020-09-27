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
#include <akari/core/logger.h>
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
            AKR_ASSERT(p != nullptr);
            AKR_ASSERT(intptr_t(p) % alignment == 0);
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) {
            // debug("deallocate {}\n", p);
            CUDA_CHECK(cudaFree(p));
        }
        bool do_is_equal(const memory_resource &other) const noexcept { return &other == this; }
    };
    class cuda_memory_resource : public astd::pmr::memory_resource {
      public:
        void *do_allocate(size_t bytes, size_t alignment) {
            void *p;
            CUDA_CHECK(cudaMalloc(&p, bytes));
            AKR_ASSERT(p != nullptr);
            AKR_ASSERT(intptr_t(p) % alignment == 0);
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) {
            // debug("deallocate {}\n", p);
            CUDA_CHECK(cudaFree(p));
        }
        bool do_is_equal(const memory_resource &other) const noexcept { return &other == this; }
    };
    static cuda_unified_memory_resource _cuda_unified_memory_resource;
    static cuda_memory_resource _cuda_memory_resource;
#endif
    namespace _mode_internal {
        static ComputeDevice cur_device = ComputeDevice::cpu;
        static astd::pmr::memory_resource *_device_memory_resource = astd::pmr::get_default_resource();
    } // namespace _mode_internal
#ifndef AKR_ENABLE_GPU
    void set_device_gpu() {
        fatal("gpu rendering is not supported\n");
        std::abort();
    }
    AKR_EXPORT void sync_device() {}
#else
    AKR_EXPORT void sync_device() { CUDA_CHECK(cudaDeviceSynchronize()); }

    void set_device_gpu() {
        _mode_internal::cur_device = ComputeDevice::gpu;
        _mode_internal::_device_memory_resource = &_cuda_unified_memory_resource;
    }
#endif
    void set_device_cpu() {
        _mode_internal::cur_device = ComputeDevice::cpu;
        _mode_internal::_device_memory_resource = astd::pmr::get_default_resource();
    }
    AKR_EXPORT ComputeDevice get_device() { return _mode_internal::cur_device; }
    astd::pmr::memory_resource *get_managed_memory_resource() { return _mode_internal::_device_memory_resource; }
    astd::pmr::memory_resource *get_device_memory_resource() { return &_cuda_memory_resource; }
} // namespace akari