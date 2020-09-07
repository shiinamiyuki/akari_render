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
    // Can allocate concurrently
    class SmallArena {
        device_ptr<uint8_t> buffer = nullptr;
        size_t size;
        std::atomic<size_t> allocated;
        static constexpr size_t align16(size_t x) { return (x + 15ULL) & (~15ULL); }

      public:
        SmallArena(SmallArena &&rhs) : buffer(rhs.buffer), size(rhs.size), allocated(rhs.allocated.load()) {
            rhs.buffer = nullptr;
            rhs.size = 0u;
            rhs.allocated = 0;
        }
        SmallArena(device_ptr<uint8_t> buffer, size_t size) : buffer(buffer), size(size), allocated(0) {}
        template <typename T, typename... Args>
        device_ptr<T> alloc(Args &&... args) {
            size_t bytes_needed = align16(sizeof(T));
            size_t cur = allocated.fetch_add(bytes_needed);
            if (cur >= size) {
                return nullptr;
            }
            device_ptr<T> p = reinterpret_cast<device_ptr<T>>(buffer + cur);
            new (p) T(std::forward<Args>(args)...);
            return p;
        }
        void reset() { allocated = 0; }
    };
} // namespace akari