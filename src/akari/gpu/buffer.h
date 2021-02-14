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
#include <memory>
namespace akari::gpu {
    class Dispatcher;
    class RawBuffer : NonCopyable {
      public:
        class Impl {
          public:
            virtual void download(Dispatcher &dispatcher, size_t offset, size_t size, void *host_data)     = 0;
            virtual void upload(Dispatcher &dispatcher, size_t offset, size_t size, const void *host_data) = 0;
            virtual size_t size() const                                                                    = 0;
            virtual void *ptr()                                                                            = 0;
            virtual ~Impl()                                                                                = default;
        };
        RawBuffer(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}
        Impl *impl_mut() const { return impl.get(); }
        void *ptr() const { return impl->ptr(); }

      protected:
        std::unique_ptr<Impl> impl;
    };
    template <class T>
    class Buffer : public RawBuffer {
      public:
        using RawBuffer::RawBuffer;
        Buffer(RawBuffer buf) : RawBuffer(std::move(buf)) {}
        T *data() const { return reinterpret_cast<T *>(ptr()); }
        size_t size() const { return impl->size() / sizeof(T); }
        void download(Dispatcher &dispatcher, size_t offset, size_t size, T *host_data) {
            AKR_ASSERT(offset * sizeof(T) + size * sizeof(T) <= impl->size());
            impl->download(dispatcher, offset * sizeof(T), size * sizeof(T), host_data);
        }
        void upload(Dispatcher &dispatcher, size_t offset, size_t size, const T *host_data) {
            AKR_ASSERT(offset * sizeof(T) + size * sizeof(T) <= impl->size());
            impl->upload(dispatcher, offset * sizeof(T), size * sizeof(T), host_data);
        }
    };
    // template <class T>
    // class BufferView {
    //     BufferView(Buffer<T> & buf){

    //     }
    //   public:
    //     Buffer<T>::I _buf;
    // };

} // namespace akari::gpu