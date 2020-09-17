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

#include <akari/common/def.h>
namespace akari {
    template <typename First, typename... Rest>
    struct TupleHelper {
        First first;
        TupleHelper<Rest...> rest;
        static constexpr bool is_null() { return false; }
    };

    struct TupleNull {
        static constexpr bool is_null() { return true; }
    };
    template <typename First>
    struct TupleHelper<First> {
        First first;
        TupleNull rest;
        static constexpr bool is_null() { return false; }
    };

    template <typename... Ts>
    struct Tuple : TupleHelper<Ts...> {
        static constexpr bool is_null() { return TupleHelper<Ts...>::is_null(); }
        template <typename T, typename F>
        static void foreach_helper_cpu(int idx, T &tuple, F f) {
            if constexpr (!T::is_null()) {
                f(idx, tuple.first);
                foreach_helper_cpu(idx + 1, tuple.rest, f);
            }
        }
        template <typename T, typename F>
        AKR_XPU static void foreach_helper(int idx, T &tuple, F f) {
            if constexpr (!T::is_null()) {
                f(idx, tuple.first);
                foreach_helper(idx + 1, tuple.rest, f);
            }
        }
        template <typename F>
        AKR_XPU void foreach (F f) {
            foreach_helper(*this, [&](int i, auto &&arg) { f(arg); });
        }
        template <typename F>
        AKR_XPU void foreach_index(F f) {
            foreach_helper(*this, f);
        }
        template <typename F>
        void foreach_cpu(F f) {
            foreach_helper_cpu(*this, [&](int i, auto &&arg) { f(arg); });
        }
    };
} // namespace akari