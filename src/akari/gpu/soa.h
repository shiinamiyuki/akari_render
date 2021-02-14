#// Copyright 2020 shiinamiyuki
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

namespace akari::gpu {
    template <class T>
    struct SOA {
        T *_ptr      = nullptr;
        size_t _size = 0;
        SOA()        = default;
        SOA(T * p, size_t N):_ptr(p),_size(N){}
        size_t size() const { return _size; }
        T &operator[](uint32_t i) { return _ptr[i]; }
        const T &operator[](uint32_t i) const { return _ptr[i]; }
    };
} // namespace akari::gpu