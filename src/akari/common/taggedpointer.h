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
#include <cstdint>
#include <akari/common/panic.h>

namespace akari {
    namespace detail {
        template <typename... T> struct TypeIndex {
            template <typename U, typename Tp, typename... Rest> struct GetIndex_ {
                static const int value = std::is_same<Tp, U>::value ? 0 : 1 + GetIndex_<U, Rest...>::value;
            };
            template <typename U, typename Tp> struct GetIndex_<U, Tp> {
                static const int value = std::is_same<Tp, U>::value ? 0 : -1;
            };
            template <int I, typename Tp, typename... Rest> struct GetType_ {
                using type = typename std::conditional<I == 0, Tp, typename GetType_<I - 1, Rest...>::type>::type;
            };

            template <int I, typename Tp> struct GetType_<I, Tp> {
                using type = typename std::conditional<I == 0, Tp, void>::type;
            };

            template <typename U> struct GetIndex { static const int value = GetIndex_<U, T...>::value; };

            template <int N> struct GetType { using type = typename GetType_<N, T...>::type; };
        };
        template <class T, class... Rest> struct FirstOf { using type = T; };

    } // namespace detail

    template <typename... Ts> struct TaggedPointer {
        static constexpr int nTypes = sizeof...(Ts);
        using Index = detail::TypeIndex<Ts...>;
        template <typename U> TaggedPointer(U *ptr) {
            uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
            static_assert(Index::template GetIndex<U>::value != -1);
            constexpr uint64_t index = Index::template GetIndex<U>::value;
            bits = (index << tag_shift) | (iptr & ptr_mask);
        }
        TaggedPointer() = default;
        template <typename U> bool isa() const {
            static_assert(Index::template GetIndex<U>::value != -1);
            constexpr uint64_t index = Index::template GetIndex<U>::value;
            return ((bits & tag_mask) >> tag_shift) == index;
        }
        void *get_ptr() const { return reinterpret_cast<void *>(ptr_mask & bits); }
        template <typename U> U *cast() const {
            if (!isa<U>()) {
                return nullptr;
            }
            return reinterpret_cast<U *>(get_ptr());
        }
        template <typename Visitor> void dispatch(Visitor &&f) const {
#define _GEN_CASE_N(N)                                                                                                 \
    case N:                                                                                                            \
        if constexpr (N < nTypes) {                                                                                    \
            using ty = typename Index::template GetType<N>::type;                                                      \
            return f(*cast<ty>());                                                                                     \
        };
#define _GEN_CASES_2()                                                                                                 \
    _GEN_CASE_N(0)                                                                                                     \
    _GEN_CASE_N(1)
#define _GEN_CASES_4()                                                                                                 \
    _GEN_CASES_2()                                                                                                     \
    _GEN_CASE_N(2)                                                                                                     \
    _GEN_CASE_N(3)
#define _GEN_CASES_8()                                                                                                 \
    _GEN_CASES_4()                                                                                                     \
    _GEN_CASE_N(4)                                                                                                     \
    _GEN_CASE_N(5)                                                                                                     \
    _GEN_CASE_N(6)                                                                                                     \
    _GEN_CASE_N(7)
#define _GEN_CASES_16()                                                                                                \
    _GEN_CASES_8()                                                                                                     \
    _GEN_CASE_N(8)                                                                                                     \
    _GEN_CASE_N(9)                                                                                                     \
    _GEN_CASE_N(10)                                                                                                    \
    _GEN_CASE_N(11)                                                                                                    \
    _GEN_CASE_N(12)                                                                                                    \
    _GEN_CASE_N(13)                                                                                                    \
    _GEN_CASE_N(14)                                                                                                    \
    _GEN_CASE_N(15)
            using Ret = std::invoke_result_t<Visitor, typename detail::FirstOf<Ts...>::type &>;
            uint64_t index = (bits & tag_mask) >> tag_shift;
            static_assert(nTypes <= 16, "too many types");
            if constexpr (nTypes <= 2) {
                switch (index) { _GEN_CASES_2(); }
            } else if constexpr (nTypes <= 4) {
                switch (index) { _GEN_CASES_4(); }
            } else if constexpr (nTypes <= 8) {
                switch (index) { _GEN_CASES_8(); }
            } else if constexpr (nTypes <= 16) {
                switch (index) { _GEN_CASES_16(); }
            }
            if constexpr (std::is_same_v<void, Ret>) {
                return;
            } else {
                AKR_PANIC("No matching case");
            }
        }

      private:
        static constexpr uint64_t tag_shift = 48u;
        static constexpr uint64_t tag_mask = (((1ull << 16) - 1) << tag_shift);
        static constexpr uint64_t ptr_mask = ~tag_mask;
        uintptr_t bits = 0;
        static_assert(sizeof(void *) == 8);
    };
} // namespace akari