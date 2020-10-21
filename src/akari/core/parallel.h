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

#ifndef AKARIRENDER_PARALLEL_HPP
#define AKARIRENDER_PARALLEL_HPP

#include <akari/core/akari.h>
#include <akari/core/math.h>
#include <atomic>
#include <functional>
#include <future>

namespace akari {
    
    class AtomicFloat {
        std::atomic<float> val;

      public:
        using Float = float;
        explicit AtomicFloat(Float v = 0) : val(v) {}

        AtomicFloat(const AtomicFloat &rhs) : val((float)rhs.val) {}

        void add(Float v) {
            auto current = val.load();
            while (!val.compare_exchange_weak(current, current + v)) {
            }
        }

        [[nodiscard]] float value() const { return val.load(); }

        explicit operator float() const { return value(); }

        void set(Float v) { val = v; }
    };

    AKR_EXPORT void parallel_for(int count, const std::function<void(uint32_t, uint32_t)> &func, size_t chunkSize = 1);
    AKR_EXPORT size_t num_work_threads();
    inline void parallel_for_2d(const ivec2 &dim, const std::function<void(ivec2, uint32_t)> &func,
                                size_t chunkSize = 1) {
        parallel_for(
            dim.x * dim.y,
            [&](uint32_t idx, int tid) {
                auto x = idx % dim.x;
                auto y = idx / dim.x;
                func(ivec2(x, y), tid);
            },
            chunkSize);
    }
    namespace thread {
        AKR_EXPORT void finalize();
    }

    // Wrapper around std::future<T>
    template <typename T>
    class Future {
        std::future<T> inner;
        template <typename R>
        friend class Future;

      public:
        Future(std::future<T> ft) : inner(std::move(ft)) {}
        template <typename F, typename R = std::invoke_result_t<F, decltype(std::declval<std::future<T>>().get())>>
        auto then(F &&f, std::launch policy = std::launch::deferred) -> Future<R> {
            return Future<R>(std::async(std::launch::deferred, [=, ft = std::move(inner)]() mutable {
                if constexpr (std::is_same_v<T, void>) {
                    ft.get();
                    return f();
                } else {
                    decltype(auto) result = ft.get();
                    return f(result);
                }
            }));
        }
    };
    template <class _Fty, class... _ArgTypes>
    Future<std::invoke_result_t<std::decay_t<_Fty>, std::decay_t<_ArgTypes>...>>
    async_do(std::launch policy, _Fty &&_Fnarg, _ArgTypes &&... _Args) {
        return std::async(policy, std::forward<_Fty>(_Fnarg), std::forward<_ArgTypes>(_Args)...);
    }
} // namespace akari
#endif // AKARIRENDER_PARALLEL_HPP
