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
#include <akari/core/astd.h>
#include <functional>
namespace akari {
    template <typename T = astd::byte>
    using Allocator = astd::pmr::polymorphic_allocator<T>;

    template <typename T>
    using Box = std::unique_ptr<T, std::function<void(T *)>>;
    template <typename T, typename... Ts>
    Box<T> make_box(Ts &&... args) {
        auto rsrc = astd::pmr::get_default_resource();
        Allocator<> alloc(rsrc);
        return Box<T>(alloc.new_object<T>(std::forward<Ts>(args)...), [=](T *p) {
            Allocator<> alloc_(rsrc);
            alloc_.destroy(p);
            alloc_.deallocate_object(const_cast<std::remove_const_t<T> *>(p), 1);
        });
    }
    template <typename T, typename... Ts>
    Box<T> make_pmr_box(Allocator<> alloc, Ts &&... args) {
        return Box<T>(alloc.new_object<T>(std::forward<Ts>(args)...), [=](T *p) mutable {
            alloc.destroy(p);
            alloc.deallocate_object(const_cast<std::remove_const_t<T> *>(p), 1);
        });
    }

    template <typename T, typename... Ts>
    std::shared_ptr<T> make_pmr_shared(Allocator<> alloc, Ts &&... args) {
        return std::shared_ptr<T>(alloc.new_object<T>(std::forward<Ts>(args)...), [=](T *p) mutable {
            alloc.destroy(p);
            alloc.deallocate_object(const_cast<std::remove_const_t<T> *>(p), 1);
        });
    }

} // namespace akari