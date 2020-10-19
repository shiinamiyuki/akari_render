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
#include <akari/core/platform.h>
#include <akari/core/buffer.h>
#include <akari/core/color.h>
namespace akari {
    template <typename T>
    struct SOA {
        SOA() = default;

        template <typename Allocator>
        SOA(int n, Allocator &&allocator) : _size(n) {
            array = allocator.template allocate_object<T>(n);
        }
         T &operator[](int i) { return array[i]; }
         const T &operator[](int i) const { return array[i]; }
         size_t size() const { return _size; }

      private:
        int _size = 0;
        T *__restrict__ array = nullptr;
    };
#if 1
    template <typename T, int N, typename A>
    struct SOAVectorXT {
        using Self = SOAVectorXT<T, N, A>;
        using value_type = A;
        static_assert(sizeof(A) % sizeof(T) == 0);
        static constexpr size_t stride = sizeof(A) / sizeof(T);
        SOAVectorXT() = default;
        template <typename Allocator>
        SOAVectorXT(int n, Allocator &&allocator) : _size(n) {
            for (int i = 0; i < N; i++)
                arrays[i] = (T *)allocator.template allocate_object<T>(n);
        }
        struct IndexHelper {
            Self &self;
            int idx;
             operator value_type() {
                value_type ret;
                for (int i = 0; i < N; i++) {
                    ret[i] = self.arrays[i][idx];
                }
                return ret;
            }
             const value_type &operator=(const value_type &rhs) {
                for (int i = 0; i < N; i++) {
                    self.arrays[i][idx] = rhs[i];
                }
                return rhs;
            }
        };
        struct ConstIndexHelper {
            const Self &self;
            int idx;
             operator value_type() const {
                value_type ret;
                for (int i = 0; i < N; i++) {
                    ret[i] = self.arrays[i][idx];
                }
                return ret;
            }
        };
         auto operator[](int idx) { return IndexHelper{*this, idx}; };
         auto operator[](int idx) const { return ConstIndexHelper{*this, idx}; };
        int _size = 0;
        T *__restrict__ arrays[N] = {nullptr};
    };

    template <int N>
    struct SOA<Vector<float, N>> : SOAVectorXT<float, N, Vector<float, N>> {
        using SOAVectorXT<float, N, Vector<float, N>>::SOAVectorXT;
    };
    template <int N>
    struct SOA<Vector<int, N>> : SOAVectorXT<int, N, Vector<int, N>> {
        using SOAVectorXT<int, N, Vector<int, N>>::SOAVectorXT;
    };
    template <int N>
    struct SOA<Color<float, N>> : SOAVectorXT<float, N, Color<float, N>> {
        using SOAVectorXT<float, N, Color<float, N>>::SOAVectorXT;
    };
#endif
} // namespace akari