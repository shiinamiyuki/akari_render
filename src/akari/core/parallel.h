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
#include <akari/core/util.h>
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
    namespace thread {
        template <int N>
        struct BlockedDim {
            static_assert(N <= 3 && N >= 1);
            Vector<int, N> dim;
            Vector<int, N> block;
        };
        template <int N>
        BlockedDim<N> blocked_range(const Vector<int, N> dim, const Vector<int, N> &block = Vector<int, N>(1)) {
            return BlockedDim<N>{dim, block};
        }
        template <int N>
        BlockedDim<N> blocked_range(const int dim, const int block) {
            return BlockedDim<N>{Vector<int, N>(dim), Vector<int, N>(block)};
        }
        AKR_EXPORT void parallel_for_impl(size_t count, const std::function<void(size_t, uint32_t)> &func,
                                          size_t chunkSize);

        inline void parallel_for(size_t count, const std::function<void(size_t, uint32_t)> &func) {
            parallel_for_impl(count, func, 1);
        }
        inline void parallel_for(BlockedDim<1> blocked_dim, const std::function<void(size_t, uint32_t)> &func) {
            parallel_for_impl(blocked_dim.dim[0], func, blocked_dim.block[0]);
        }
        AKR_EXPORT size_t num_work_threads();
        inline void parallel_for(BlockedDim<2> blocked_dim, const std::function<void(ivec2, uint32_t)> &func) {
            ivec2 tiles = (blocked_dim.dim + blocked_dim.block - ivec2(1)) / blocked_dim.block;
            parallel_for(tiles.x * tiles.y, [&](size_t idx, int tid) {
                ivec2 t(idx % tiles.x, idx / tiles.x);
                for (int ty = 0; ty < blocked_dim.block[1]; ty++) {
                    for (int tx = 0; tx < blocked_dim.block[0]; tx++) {
                        int x = tx + t[0] * blocked_dim.block[0];
                        int y = ty + t[1] * blocked_dim.block[1];
                        func(ivec2(x, y), tid);
                    }
                }
            });
        }
        inline void parallel_for(BlockedDim<3> blocked_dim, const std::function<void(ivec3, uint32_t)> &func) {
            ivec3 tiles = (blocked_dim.dim + blocked_dim.block - ivec3(1)) / blocked_dim.block;
            parallel_for(tiles.x * tiles.y * tiles.z, [&](size_t idx, int tid) {
                auto z_ = idx / (tiles.x * tiles.y);
                auto y_ = (idx % (tiles.x * tiles.y)) / tiles.x;
                auto x_ = (idx % (tiles.x * tiles.y)) % tiles.x;
                ivec3 t(x_, y_, z_);
                for (int tz = 0; tz < blocked_dim.block[2]; tz++) {
                    for (int ty = 0; ty < blocked_dim.block[1]; ty++) {
                        for (int tx = 0; tx < blocked_dim.block[0]; tx++) {
                            int x = tx + t[0] * blocked_dim.block[0];
                            int y = ty + t[1] * blocked_dim.block[1];
                            int z = tz + t[2] * blocked_dim.block[2];
                            func(ivec3(x, y, z), tid);
                        }
                    }
                }
            });
        }
        template <class ParIter, class F>
        auto parallel_for_each(ParIter begin, ParIter end, F &&f) -> decltype(*std::declval<ParIter>()) {
            size_t count = std::distance(begin, end);
            parallel_for(count, [&](size_t idx, uint32_t) { f(*(begin + idx)); });
        }
        // F :: T -> size_t -> T
        template <class ParIter, class T, class F>
        T parallel_reduce(ParIter begin, ParIter end, T init, F &&f) {
            T acc = init;
            std::mutex m;
            size_t count = std::distance(begin, end);
            auto block_size = (count + thread::num_work_threads() - 1) / num_work_threads();
            parallel_for(thread::num_work_threads(), [&](size_t idx, uint32_t tid) {
                T cache = init;
                auto it = begin + tid * block_size;
                for (size_t i = 0; i < block_size && it < end; i++) {
                    cache = f(cache, *);
                    it++;
                }
                std::lock_guard<std::mutex> lock(m);
                acc = f(acc, cache);
            });
        }
        AKR_EXPORT void init(size_t num_threads);
        AKR_EXPORT void finalize();
    } // namespace thread

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
    async_do(std::launch policy, _Fty &&_Fnarg, _ArgTypes &&..._Args) {
        return std::async(policy, std::forward<_Fty>(_Fnarg), std::forward<_ArgTypes>(_Args)...);
    }
} // namespace akari
#endif // AKARIRENDER_PARALLEL_HPP
