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
#include <vector>
#include <memory_resource>
#include <akari/common/fwd.h>
#include <akari/common/buffer.h>
namespace akari {
    template <typename T>
    struct Box {
        using Allocator = astd::pmr::polymorphic_allocator<T>;
        Box(MemoryResource *resource) : allocator(resource) {}
        Box(const Box &) = delete;
        Box(Box &&rhs) : _ptr(rhs._ptr), allocator(std::move(rhs.allocator)) { rhs._ptr = nullptr; }
        Box &operator=(const Box &) = delete;
        Box &operator=(Box &&rhs) {
            reset();
            allocator = std::move(rhs.allocator);
            _ptr = rhs._ptr;
            rhs._ptr = nullptr;
            return *this;
        }
        template <typename... Ts>
        static Box make(MemoryResource *resource, Ts &&... args) {
            Box<T> box(resource);
            box.emplace(std::forward<Ts>(args)...);
            return box;
        }
        template <typename... Ts>
        void emplace(Ts &&... args) {
            reset();
            _ptr = allocator.allocate(1);
            AKR_ASSERT(_ptr);
            allocator.construct(_ptr, std::forward<Ts>(args)...);
        }
        void reset() {
            if (_ptr) {
                allocator.destroy(_ptr);
                allocator.deallocate(_ptr, 1);
            }
            _ptr = nullptr;
        }
        AKR_XPU T &operator*() const { return *_ptr; }
        AKR_XPU T *get() const { return _ptr; }
        AKR_XPU T *operator->() const { return _ptr; }
        AKR_XPU operator bool() const { return _ptr != nullptr; }
        ~Box() { reset(); }

      private:
        T *_ptr = nullptr;
        Allocator allocator;
    };
} // namespace akari
