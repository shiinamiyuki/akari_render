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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <akari/gpu/buffer.h>
#include <akari/gpu/dispatch.h>
#include <akari/gpu/device.h>
#include <spdlog/spdlog.h>
#include <akari/gpu/cuda/check.h>

namespace akari::gpu {
    class CUDADispatcher : public Dispatcher::Impl {
      public:
        cudaStream_t stream;
        CUDADispatcher(cudaStream_t stream) : stream(stream) {}
        void wait() override { CUDA_CHECK(cudaStreamSynchronize(stream)); }
        void then(std::function<void(void)> F) override {
            using Func   = std::function<void(void)>;
            Func *f      = new Func(std::move(F));
            auto wrapper = [](void *p) {
                auto f = reinterpret_cast<Func *>(p);
                (*f)();
                delete f;
            };
            cudaLaunchHostFunc(stream, wrapper, (void *)f);
        }
        ~CUDADispatcher() { CUDA_CHECK(cudaStreamDestroy(stream)); }
    };
    class CUDABuffer : public RawBuffer::Impl {
        void *ptr;
        size_t bytes;

      public:
        void *device_ptr() { return ptr; }
        CUDABuffer(void *ptr, size_t bytes) : ptr(ptr), bytes(bytes) {}
        ~CUDABuffer() { CUDA_CHECK(cudaFree(ptr)); }
        size_t size() const override { return bytes; }
        void download(Dispatcher &dispatcher, size_t offset, size_t size, void *host_data) {
            auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
            CUDA_CHECK(cudaMemcpyAsync(host_data, (const uint8_t *)ptr + offset, size, cudaMemcpyDeviceToHost, stream));
        }
        void upload(Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) {
            auto stream = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
            CUDA_CHECK(cudaMemcpyAsync((uint8_t *)ptr + offset, host_data, size, cudaMemcpyHostToDevice, stream));
        }
    };
    class CUDAKernel : public Kernel::Impl {
        CUfunction func;

      public:
        CUDAKernel(CUfunction func) : func(func) {}
        void launch(Dispatcher &dispatcher, uvec3 global_size, uvec3 local_size, std::vector<void *> args) override {
            auto stream    = dynamic_cast<CUDADispatcher *>(dispatcher.impl_mut())->stream;
            vec3 grid_size = (global_size + local_size - uvec3(1)) / local_size;
            CU_CHECK(cuLaunchKernel(func, grid_size.x, grid_size.y, grid_size.z, local_size.x, local_size.y,
                                    local_size.z, 1024, stream, args.data(), nullptr));
        }
    };
    class CUDADevice : public Device::Impl {
      public:
        RawBuffer allocate_buffer(size_t bytes) override {
            void *ptr;
            CUDA_CHECK(cudaMalloc(&ptr, bytes));
            return RawBuffer(std::make_unique<CUDABuffer>(ptr, bytes));
        }
        Dispatcher new_dispatcher() override {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            return Dispatcher(std::make_unique<CUDADispatcher>(stream));
        }
    };
    inline std::shared_ptr<Device> create_cuda_device() {
        return std::make_shared<Device>(std::make_unique<CUDADevice>());
    }
} // namespace akari::gpu