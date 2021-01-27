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
#include <akari/util.h>
#include <akari/gpu/buffer.h>
#include <akari/gpu/dispatch.h>
#include <memory>
namespace akari::gpu {
    class Device {
      public:
        class Impl {
          public:
            virtual RawBuffer allocate_buffer(size_t bytes) = 0;
            virtual Dispatcher new_dispatcher()             = 0;
        };
        template <typename T>
        Buffer<T> allocate_buffer(size_t n_elements) {
            return Buffer<T>(impl->allocate_buffer(n_elements * sizeof(T)));
        }
        Dispatcher new_dispatcher() { return impl->new_dispatcher(); }

      protected:
        std::unique_ptr<Impl> impl;
    };
    std::shared_ptr<Device> create_cuda_device();
    // template <class T>
    // class BufferView {
    //     BufferView(Buffer<T> & buf){

    //     }
    //   public:
    //     Buffer<T>::I _buf;
    // };

} // namespace akari::gpu