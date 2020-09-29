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
#include <type_traits>
#include <akari/common/platform.h>
#include <akari/common/buffer.h>
#include <akari/common/color.h>
namespace akari {
    template <typename T>
    struct SOA {
        SOA() = default;

        template <typename Allocator>
        SOA(int n, Allocator &&allocator) : _size(n) {
            array = allocator.template allocate_object<T>(n);
        }
        AKR_XPU T &operator[](int i) { return array[i]; }
        AKR_XPU const T &operator[](int i) const { return array[i]; }
        AKR_XPU size_t size() const { return _size; }

      private:
        int _size = 0;
        T *__restrict__ array = nullptr;
    };

    template <typename T, int N, typename A>
    struct SOAArrayXT {
        using Self = SOAArrayXT<T, N, A>;
        using value_type = A;
        static_assert(sizeof(A) % sizeof(T) == 0);
        static constexpr size_t stride = sizeof(A) / sizeof(T);
        SOAArrayXT() = default;
        template <typename Allocator>
        SOAArrayXT(int n, Allocator &&allocator) : _size(n) {
            array = (T *)allocator.template allocate_object<A>(n);
        }
        AKR_XPU A &operator[](int idx) { return *reinterpret_cast<A *>(&array[idx * stride]); };
        AKR_XPU const A &operator[](int idx) const { return *reinterpret_cast<const A *>(&array[idx * stride]); };
        int _size = 0;
        T *__restrict__ array = nullptr;
    };
    template <int N>
    struct SOA<Array<float, N>> : SOAArrayXT<float, N, Array<float, N>> {
        using SOAArrayXT<float, N, Array<float, N>>::SOAArrayXT;
    };
    template <int N>
    struct SOA<Array<int, N>> : SOAArrayXT<int, N, Array<int, N>> {
        using SOAArrayXT<int, N, Array<int, N>>::SOAArrayXT;
    };
    template <int N>
    struct SOA<Color<float, N>> : SOAArrayXT<float, N, Color<float, N>> {
        using SOAArrayXT<float, N, Color<float, N>>::SOAArrayXT;
    };
} // namespace akari