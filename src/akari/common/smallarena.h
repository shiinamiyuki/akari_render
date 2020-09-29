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
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <akari/common/fwd.h>

namespace akari {
    // A very fast memory arena for device
    class SmallArena {
        astd::byte *buffer = nullptr;
        size_t size;
        size_t allocated = 0;
        template <size_t alignment>
        static constexpr size_t align(size_t x) {
            static_assert((alignment & (alignment - 1)) == 0);
            return (x + alignment - 1) & (~(alignment - 1));
        }

      public:
        AKR_XPU SmallArena(const SmallArena &) = delete;
        AKR_XPU SmallArena(SmallArena &&rhs) : buffer(rhs.buffer), size(rhs.size), allocated(rhs.allocated) {
            rhs.buffer = nullptr;
            rhs.size = 0u;
            rhs.allocated = 0;
        }
        AKR_XPU SmallArena(astd::byte *buffer, size_t size) : buffer(buffer), size(size), allocated(0) {}
        template <typename T, typename... Args>
        AKR_XPU T *alloc(Args &&... args) {
            size_t bytes_needed = align<alignof(T)>(sizeof(T) + alignof(T));
            // size_t cur = allocated.fetch_add(bytes_needed);
            size_t cur = allocated;
            allocated += (bytes_needed);
            if (cur >= size) {
                return nullptr;
            }
            auto q = (void *)(buffer + cur);
            AKR_ASSERT(astd::align(alignof(T), sizeof(T), q, bytes_needed));
            auto p = reinterpret_cast<T *>(q);
            new (p) T(args...);
            return p;
        }
        void reset() { allocated = 0; }
    };
} // namespace akari