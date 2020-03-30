// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#ifndef AKARIRENDER_PARALLEL_HPP
#define AKARIRENDER_PARALLEL_HPP

#include <Akari/Core/Config.h>
#include <Akari/Core/Math.h>
#include <atomic>
#include <functional>

namespace Akari {
    class AtomicFloat {
        std::atomic<float> val;
      public:
        explicit AtomicFloat(Float v = 0) : val(v) {}

        AtomicFloat(const AtomicFloat &rhs) : val((float)rhs.val) {}

        void add(Float v) {
            auto current = val.load();
            while (!val.compare_exchange_weak(current, current + v)){}
        }

        [[nodiscard]] float value() const {
            return val.load();
        }

        explicit operator float() const {
            return value();
        }

        void set(Float v){
            val = v;
        }
    };

    AKR_EXPORT void ParallelFor(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunkSize = 1);

    inline void ParallelFor2D(const ivec2 &dim, const std::function<void(ivec2, uint32_t)> &func,
                              size_t chunkSize = 1) {
        ParallelFor(
            dim.x * dim.y,
            [&](uint32_t idx, int tid) {
                auto x = idx % dim.x;
                auto y = idx / dim.x;
                func(ivec2(x, y), tid);
            },
            chunkSize);
    }

    AKR_EXPORT void ThreadPoolFinalize();

} // namespace Akari
#endif // AKARIRENDER_PARALLEL_HPP
