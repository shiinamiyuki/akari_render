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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
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
#include <algorithm>
#include <cstring>
#include <akari/common/panic.h>
#include <akari/common/platform.h>
namespace akari {
    template <typename... T>
    struct TypeIndex {
        template <typename U, typename Tp, typename... Rest>
        struct GetIndex_ {
            static const int value = std::is_same<Tp, U>::value ? 0 : 1 + GetIndex_<U, Rest...>::value;
        };
        template <typename U, typename Tp>
        struct GetIndex_<U, Tp> {
            static const int value = std::is_same<Tp, U>::value ? 0 : -1;
        };
        template <int I, typename Tp, typename... Rest>
        struct GetType_ {
            using type = typename std::conditional<I == 0, Tp, typename GetType_<I - 1, Rest...>::type>::type;
        };

        template <int I, typename Tp>
        struct GetType_<I, Tp> {
            using type = typename std::conditional<I == 0, Tp, void>::type;
        };

        template <typename U>
        struct GetIndex {
            static const int value = GetIndex_<U, T...>::value;
        };

        template <int N>
        struct GetType {
            using type = typename GetType_<N, T...>::type;
        };
    };
    template <class T, class... Rest>
    struct FirstOf {
        using type = T;
    };

    template <typename U, typename... T>
    struct SizeOf {
        static constexpr int value = std::max<int>(sizeof(U), SizeOf<T...>::value);
    };
    template <typename T>
    struct SizeOf<T> {
        static constexpr int value = sizeof(T);
    };

    template <typename... T>
    struct Variant {
      private:
        static constexpr int nTypes = sizeof...(T);
        static constexpr std::size_t alignment_value = std::max({alignof(T)...});
        typename std::aligned_storage<SizeOf<T...>::value, alignment_value>::type data;
        int index = -1;

      public:
        using Index = TypeIndex<T...>;

        Variant() = default;

        template <typename U, typename = std::enable_if_t<Index::template GetIndex<U>::value != -1, void>>
        AKR_XPU Variant(const U &u) {
            new (&data) U(u);
            index = Index::template GetIndex<U>::value;
        }

        AKR_XPU Variant(const Variant &v) : index(v.index) {
            v.dispatch([&](const auto &item) {
                using U = std::decay_t<decltype(item)>;
                new (&data) U(item);
            });
        }
        AKR_XPU int typeindex() const { return index; }
        template <typename U>
        AKR_XPU int indexof() const {
            return Index::template GetIndex<U>::value;
        }
        AKR_XPU Variant &operator=(const Variant &v) noexcept {
            if (this == &v)
                return *this;
            if (index != -1)
                _drop();
            index = v.index;
            auto that = this;
            v.dispatch([&](const auto &item) {
                using U = std::decay_t<decltype(item)>;
                *that->template get<U>() = item;
            });
            return *this;
        }

        AKR_XPU Variant(Variant &&v) noexcept : index(v.index) {
            index = v.index;
            v.index = -1;
            std::memcpy(&data, &v.data, sizeof(data));
        }

        AKR_XPU Variant &operator=(Variant &&v) noexcept {
            if (index != -1)
                _drop();
            index = v.index;
            v.index = -1;
            std::memcpy(&data, &v.data, sizeof(data));
            return *this;
        }

        template <typename U>
        AKR_XPU Variant &operator=(const U &u) {
            if (index != -1) {
                _drop();
            }
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            new (&data) U(u);
            index = Index::template GetIndex<U>::value;
            return *this;
        }
        template <typename U>
        AKR_XPU bool isa() const {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value == index;
        }
        template <typename U>
        AKR_XPU U *get() {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<U *>(&data);
        }

        template <typename U>
        AKR_XPU const U *get() const {
            static_assert(Index::template GetIndex<U>::value != -1, "U is not in T...");
            return Index::template GetIndex<U>::value != index ? nullptr : reinterpret_cast<const U *>(&data);
        }

#define _GEN_CASE_N(N)                                                                                                 \
    case N:                                                                                                            \
        if constexpr (N < nTypes) {                                                                                    \
            using ty = typename Index::template GetType<N>::type;                                                      \
            if constexpr (std::is_const_v<std::remove_pointer_t<decltype(this)>>)                                      \
                return visitor(*reinterpret_cast<const ty *>(&data));                                                  \
            else                                                                                                       \
                return visitor(*reinterpret_cast<ty *>(&data));                                                        \
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
#define _GEN_DISPATCH_BODY()                                                                                           \
    using Ret = std::invoke_result_t<Visitor, typename FirstOf<T...>::type &>;                                         \
    static_assert(nTypes <= 16, "too many types");                                                                     \
    if constexpr (nTypes <= 2) {                                                                                       \
        switch (index) { _GEN_CASES_2(); }                                                                             \
    } else if constexpr (nTypes <= 4) {                                                                                \
        switch (index) { _GEN_CASES_4(); }                                                                             \
    } else if constexpr (nTypes <= 8) {                                                                                \
        switch (index) { _GEN_CASES_8(); }                                                                             \
    } else if constexpr (nTypes <= 16) {                                                                               \
        switch (index) { _GEN_CASES_16(); }                                                                            \
    }                                                                                                                  \
    if constexpr (std::is_same_v<void, Ret>) {                                                                         \
        return;                                                                                                        \
    } else {                                                                                                           \
        AKR_PANIC("No matching case");                                                                                 \
    }
        template <class Visitor>
        AKR_XPU auto dispatch(Visitor &&visitor) {
            _GEN_DISPATCH_BODY()
        }

        template <class Visitor>
        AKR_XPU auto dispatch(Visitor &&visitor) const {
            _GEN_DISPATCH_BODY()
        }

        template <class Visitor>
        AKR_CPU auto dispatch_cpu(Visitor &&visitor) {
            _GEN_DISPATCH_BODY()
        }

        template <class Visitor>
        AKR_CPU auto dispatch_cpu(Visitor &&visitor) const {_GEN_DISPATCH_BODY()}

        AKR_XPU ~Variant() {
            if (index != -1)
                _drop();
        }

      private:
        AKR_XPU void _drop() {
            auto *that = this; // prevent gcc ICE
            dispatch([=](auto &&self) {
                using U = std::decay_t<decltype(self)>;
                that->template get<U>()->~U();
            });
        }
#undef _GEN_CASE_N
#define AKR_VAR_DISPATCH(method, ...)                                                                                  \
    return this->dispatch([&, this](auto &&self) {                                                                     \
        (void)this;                                                                                                    \
        return self.method(__VA_ARGS__);                                                                               \
    });
    };

} // namespace akari