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
        void then(const std::function<void(void)> &F) override { F(); }
    };
    class CUDABufferImpl : public RawBuffer::Impl {
        void *ptr;
        size_t bytes;

      public:
        CUDABufferImpl(void *ptr, size_t bytes) : ptr(ptr), bytes(bytes) {}
        ~CUDABufferImpl() { CUDA_CHECK(cudaFree(ptr)); }
        size_t size()const override{
          return bytes;
        }
        void download(Dispatcher &dispatcher, size_t offset, size_t size, void *host_data) {
            CUDA_CHECK(cudaMemcpy(host_data, (const uint8_t *)ptr + offset, size, cudaMemcpyDeviceToHost));
        }
        void upload(Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) {
            CUDA_CHECK(cudaMemcpy((uint8_t *)ptr + offset, host_data, size, cudaMemcpyHostToDevice));
        }
    };
    class CUDADevice : public Device::Impl {
      public:
        RawBuffer allocate_buffer(size_t bytes) override {
            void *ptr;
            CUDA_CHECK(cudaMalloc(&ptr, bytes));
            return RawBuffer(std::make_unique<CUDABufferImpl>(ptr, bytes));
        }
        Dispatcher new_dispatcher() override { return Dispatcher(std::make_unique<CUDADispatcher>()); }
    };
} // namespace akari::gpu