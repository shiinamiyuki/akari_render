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

#include "device.h"
#include <akari/core/logger.h>
#ifdef AKR_ENABLE_GPU
#    include <cuda.h>
#    include <cuda_runtime_api.h>
#    include <akari/kernel/cuda/util.h>
#endif

namespace akari {
    template <typename Base>
    class CPUResourceBase : public Base {
        astd::pmr::memory_resource *resource;

      public:
        CPUResourceBase(Device *device) : Base(device) { resource = astd::pmr::get_default_resource(); }
        void *do_allocate(size_t bytes, size_t alignment) {
            auto p = resource->allocate(bytes, alignment);
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) { resource->deallocate(p, bytes, alignment); }
        bool do_is_equal(const astd::pmr::memory_resource &other) const noexcept { return &other == this; }
    };
    class CPUDevice : public Device {
        CPUResourceBase<HostMemoryResource> host;
        CPUResourceBase<DeviceMemoryResource> device;
        CPUResourceBase<ManagedMemoryResource> managed;

      public:
        CPUDevice() : host(this), device(this), managed(this) {}
        HostMemoryResource *host_resource() override { return &host; }
        DeviceMemoryResource *device_resource() override { return &device; }
        ManagedMemoryResource *managed_resource() override { return &managed; }
        void copy_host_to_device_async(void *device_ptr, const void *host_ptr, size_t bytes) override {
            copy_host_to_device(device_ptr, host_ptr, bytes);
        }
        void copy_device_to_host_async(void *host_ptr, const void *device_ptr, size_t bytes) override {
            copy_device_to_host(host_ptr, device_ptr, bytes);
        }
        void copy_host_to_device(void *device_ptr, const void *host_ptr, size_t bytes) override {
            std::memcpy(device_ptr, host_ptr, bytes);
        }
        void copy_device_to_host(void *host_ptr, const void *device_ptr, size_t bytes) override {
            std::memcpy(host_ptr, device_ptr, bytes);
        }
        void prefetch_managed(void *p, size_t bytes) {}
    };

#ifdef AKR_ENABLE_GPU
    class cuda_unified_memory_resource : public ManagedMemoryResource {
      public:
        using ManagedMemoryResource::ManagedMemoryResource;
        void *do_allocate(size_t bytes, size_t alignment) {
            void *p;
            CUDA_CHECK(cudaMallocManaged(&p, bytes));
            AKR_ASSERT(p != nullptr);
            AKR_ASSERT(intptr_t(p) % alignment == 0);
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) { CUDA_CHECK(cudaFree(p)); }
        bool do_is_equal(const memory_resource &other) const noexcept { return &other == this; }
    };
    class cuda_host_memory_resource : public HostMemoryResource {
      public:
        using HostMemoryResource::HostMemoryResource;
        void *do_allocate(size_t bytes, size_t alignment) {
            void *p;
            CUDA_CHECK(cudaMallocHost(&p, bytes));
            AKR_ASSERT(p != nullptr);
            AKR_ASSERT(intptr_t(p) % alignment == 0);
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) { CUDA_CHECK(cudaFreeHost(p)); }
        bool do_is_equal(const memory_resource &other) const noexcept { return &other == this; }
    };
    class cuda_memory_resource : public DeviceMemoryResource {
      public:
        using DeviceMemoryResource::DeviceMemoryResource;
        void *do_allocate(size_t bytes, size_t alignment) {
            void *p;
            CUDA_CHECK(cudaMalloc(&p, bytes));
            AKR_ASSERT(p != nullptr);
            AKR_ASSERT(intptr_t(p) % alignment == 0);
            return p;
        }
        void do_deallocate(void *p, size_t bytes, size_t alignment) { CUDA_CHECK(cudaFree(p)); }
        bool do_is_equal(const memory_resource &other) const noexcept { return &other == this; }
    };

    class GPUDevice : public Device {
        cuda_unified_memory_resource _cuda_unified_memory_resource;
        cuda_memory_resource _cuda_memory_resource;
        cuda_host_memory_resource _cuda_host_memory_resource;
        int cuda_device;

      public:
        GPUDevice()
            : _cuda_unified_memory_resource(this), _cuda_memory_resource(this), _cuda_host_memory_resource(this) {
            cudaFree(nullptr);
            CUDA_CHECK(cudaGetDevice(&cuda_device));
            int driverVersion;
            CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
            int runtimeVersion;
            CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
            auto versionToString = [](int version) {
                int major = version / 1000;
                int minor = (version - major * 1000) / 10;
                return fmt::format("{}.{}", major, minor);
            };
            verbose("GPU CUDA driver {}, CUDA runtime {}", versionToString(driverVersion),
                    versionToString(runtimeVersion));
            int nDevices;
            CUDA_CHECK(cudaGetDeviceCount(&nDevices));
            cudaDeviceProp firstDeviceProperties;
            std::string devices;
            for (int i = 0; i < nDevices; ++i) {
                cudaDeviceProp deviceProperties;
                CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, i));
                if (i == 0)
                    firstDeviceProperties = deviceProperties;
                AKR_CHECK(deviceProperties.canMapHostMemory);

                std::string deviceString =
                    fmt::format("CUDA device {} ({}) with {} MiB, {} SMs running at {} MHz "
                                "with shader model {}.{}",
                                i, deviceProperties.name, deviceProperties.totalGlobalMem / (1024. * 1024.),
                                deviceProperties.multiProcessorCount, deviceProperties.clockRate / 1000.,
                                deviceProperties.major, deviceProperties.minor);
                verbose("{}", deviceString);
                devices += deviceString + "\n";
            }

            int hasUnifiedAddressing;
            CUDA_CHECK(cudaDeviceGetAttribute(&hasUnifiedAddressing, cudaDevAttrUnifiedAddressing, cuda_device));
            if (!hasUnifiedAddressing) {
                fatal("The selected GPU device ({}) does not support unified addressing.", cuda_device);
                std::exit(1);
            }

            CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
            size_t stackSize;
            CUDA_CHECK(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
            verbose("Reset stack size to {}", stackSize);

            CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32 * 1024 * 1024));

            CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
        }
        HostMemoryResource *host_resource() override { return &_cuda_host_memory_resource; }
        DeviceMemoryResource *device_resource() override { return &_cuda_memory_resource; }
        ManagedMemoryResource *managed_resource() override { return &_cuda_unified_memory_resource; }
        void copy_host_to_device(void *device_ptr, const void *host_ptr, size_t bytes) override {
            CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, bytes, cudaMemcpyKind::cudaMemcpyDefault));
        }
        void copy_device_to_host(void *host_ptr, const void *device_ptr, size_t bytes) override {
            CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, bytes, cudaMemcpyKind::cudaMemcpyDefault));
        }
        void copy_host_to_device_async(void *device_ptr, const void *host_ptr, size_t bytes) override {
            CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_ptr, bytes, cudaMemcpyKind::cudaMemcpyDefault));
        }
        void copy_device_to_host_async(void *host_ptr, const void *device_ptr, size_t bytes) override {
            CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, bytes, cudaMemcpyKind::cudaMemcpyDefault));
        }
        void prefetch_managed(void *p, size_t bytes) {
            //  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, 0));
        }
        void sync() { CUDA_CHECK(cudaDeviceSynchronize()); }
    };
    AKR_EXPORT Device *gpu_device() {
        static std::once_flag flag;
        static std::unique_ptr<GPUDevice> device;
        std::call_once(flag, [&]() { device.reset(new GPUDevice()); });
        return device.get();
    }
#else
    AKR_EXPORT Device *gpu_device() {
        fatal("gpu is not supported");
        std::exit(1);
        return nullptr;
    }
#endif
    AKR_EXPORT Device *cpu_device() {
        static std::once_flag flag;
        static std::unique_ptr<CPUDevice> device;
        std::call_once(flag, [&]() { device.reset(new CPUDevice()); });
        return device.get();
    }
    static Device *___active_device = nullptr;
    AKR_EXPORT void set_device_gpu() { ___active_device = gpu_device(); }
    AKR_EXPORT void set_device_cpu() { ___active_device = cpu_device(); }
    AKR_EXPORT Device *active_device() { return ___active_device; }

} // namespace akari