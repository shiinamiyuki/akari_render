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
namespace akari::gpu {
    class Dispatcher : NonCopyable {
      public:
        class Impl {
          public:
            virtual ~Impl()                                = default;
            virtual void then(std::function<void(void)> F) = 0;
            virtual void wait()                            = 0;
        };
        void wait() { impl->wait(); }
        Dispatcher &then(std::function<void(void)> F) {
            impl->then(std::move(F));
            return *this;
        }
        Dispatcher(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}
        Impl *impl_mut() const { return impl.get(); }

      protected:
        std::unique_ptr<Impl> impl;
    };
} // namespace akari::gpu