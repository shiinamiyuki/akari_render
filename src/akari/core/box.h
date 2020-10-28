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
#include <akari/core/memory.h>
#include <functional>
namespace akari {
    template <typename T>
    struct Box {
        using Deleter = std::function<void(T *)>;
        Box(Deleter deleter) : deleter(deleter) {}
        Box(T *ptr, Deleter deleter) : _ptr(ptr), deleter(deleter) {}
        Box(const Box &) = delete;
        Box(Box &&rhs) : _ptr(rhs._ptr), deleter(std::move(rhs.deleter)) { rhs._ptr = nullptr; }
        Box &operator=(const Box &) = delete;
        Box &operator=(Box &&rhs) {
            reset();
            deleter = std::move(rhs.deleter);
            _ptr = rhs._ptr;
            rhs._ptr = nullptr;
            return *this;
        }
        void reset() {
            if (_ptr) {
                deleter(_ptr);
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
        Deleter deleter;
    };
    template <typename T, typename... Ts>
    Box<T> make_pmr_box(Ts &&... args) {
        auto rsrc = astd::pmr::get_default_resource();
        Allocator alloc(rsrc);
        return Box<T>(alloc.new_object<T>(std::forward<Ts>(args)...), [=](T *p) {
            Allocator alloc_(rsrc);
            alloc_.destroy(p);
            alloc_.deallocate(p, 1);
        });
    }
} // namespace akari