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

#include <array>
#include <cstdint>
#include <immintrin.h>
#include <type_traits>
#include <xmmintrin.h>
constexpr size_t ptr_size = sizeof(void *);
template <size_t N, size_t Align = ptr_size> constexpr size_t align_to = ((N + Align - 1u) / Align) * Align;

template <typename T, size_t N> struct array_alignment { constexpr static size_t value = align_to<sizeof(T)>; };
template <typename T, size_t N> struct array_padded_size { constexpr static size_t value = N; };
template <size_t N> struct array_padded_size<float, N> { constexpr static size_t value = N < 2 ? N : align_to<N, 4>; };
template <size_t N> struct array_padded_size<int32_t, N> {
    constexpr static size_t value = N < 2 ? N : align_to<N, 4>;
};
template <typename T, size_t N> struct array_storage {
    alignas(array_alignment<T, N>::value) T data[array_padded_size<T, N>::value];
};
template <size_t N> struct array_mask;

template <size_t N> bool any(const array_mask<N> &mask) {
    bool result = false;
    for (size_t i = 0; i < N; i++) {
        result = result || mask[i];
    }
    return result;
}

template <size_t N> bool all(const array_mask<N> &mask) {
    bool result = true;
    for (size_t i = 0; i < N; i++) {
        result = result && mask[i];
    }
    return result;
}

template <size_t N> struct array_mask : std::array<int32_t, array_padded_size<int32_t, N>::value> {
    operator bool() const { return any(*this); }
};

template <typename T, size_t N> struct simd_array : public array_storage<T, N> {
    T &operator[](size_t i) { return this->data[i]; }
    const T &operator[](size_t i) const { return this->data[i]; }
};

// template <typename T, size_t N> struct masked_simd_array : simd_array<T, N> { array_mask<N> mask; };

#define AKR_SIMD_DEFAULT_OPERATOR(delegate, assign_op)                                                                 \
    template <typename T, size_t N> struct delegate {                                                                  \
        using A = simd_array<T, N>;                                                                                    \
        static void apply(A &dst, const A &src) {                                                                      \
            for (size_t i = 0; i < N; i++) {                                                                           \
                dst[i] assign_op src[i];                                                                               \
            }                                                                                                          \
        }                                                                                                              \
    };

#define AKR_SIMD_DEFAULT_SCALAR_OPERATOR(delegate, assign_op)                                                          \
    template <typename T, size_t N> struct delegate {                                                                  \
        using A = simd_array<T, N>;                                                                                    \
        static void apply(A &dst, const T &src) {                                                                      \
            for (size_t i = 0; i < N; i++) {                                                                           \
                dst[i] assign_op src;                                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    };

template <typename T, size_t N> struct array_select {
    using A = simd_array<T, N>;
    static A apply(const array_mask<N> &mask, const A &a, const T &b) {
        A tmp;
        for (size_t i = 0; i < N; i++) {
            if (mask[i]) {
                tmp[i] = b[i];
            } else {
                tmp[i] = a[i];
            }
        }
    }
};

AKR_SIMD_DEFAULT_OPERATOR(array_operator_add, +=)
AKR_SIMD_DEFAULT_OPERATOR(array_operator_sub, -=)
AKR_SIMD_DEFAULT_OPERATOR(array_operator_mul, *=)
AKR_SIMD_DEFAULT_OPERATOR(array_operator_div, /=)
AKR_SIMD_DEFAULT_SCALAR_OPERATOR(array_operator_scalar_mul, *=)
AKR_SIMD_DEFAULT_SCALAR_OPERATOR(array_operator_scalar_div, /=)

#define AKR_SIMD_GEN_OPERATOR_(op, delegate)                                                                           \
    template <typename T, typename U, size_t N>                                                                        \
    inline auto operator op(const simd_array<T, N> &a, const simd_array<U, N> &b) {                                    \
        using ResultElemT = decltype(std::declval<T>() + std::declval<U>());                                           \
        static_assert(std::is_same_v<T, ResultElemT> || std::is_same_v<U, ResultElemT>, "What ?");                     \
        if constexpr (std::is_same_v<T, ResultElemT>) {                                                                \
            auto tmpA = a;                                                                                             \
            auto tmpB = simd_array<ResultElemT, N>(b);                                                                 \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        } else {                                                                                                       \
            auto tmpA = simd_array<ResultElemT, N>(b);                                                                 \
            auto &tmpB = b;                                                                                            \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        }                                                                                                              \
    }

#define AKR_SIMD_GEN_SCALAR_RIGHT_OPERATOR_(op, delegate)                                                              \
    template <typename T, typename U, size_t N> inline auto operator op(const simd_array<T, N> &a, const U &b) {       \
        using ResultElemT = decltype(std::declval<T>() + std::declval<U>());                                           \
        static_assert(std::is_same_v<T, ResultElemT> || std::is_same_v<U, ResultElemT>, "What ?");                     \
        if constexpr (std::is_same_v<T, ResultElemT>) {                                                                \
            auto tmpA = a;                                                                                             \
            auto tmpB = T(b);                                                                                          \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        } else {                                                                                                       \
            auto tmpA = simd_array<ResultElemT, N>(b);                                                                 \
            auto &tmpB = b;                                                                                            \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        }                                                                                                              \
    }

#define AKR_SIMD_GEN_SCALAR_LEFT_OPERATOR_(op, delegate)                                                               \
    template <typename T, typename U, size_t N> inline auto operator op(const T &a, const simd_array<U, N> &b) {       \
        using ResultElemT = decltype(std::declval<T>() + std::declval<U>());                                           \
        static_assert(std::is_same_v<T, ResultElemT> || std::is_same_v<U, ResultElemT>, "What ?");                     \
        if constexpr (std::is_same_v<T, ResultElemT>) {                                                                \
            auto tmpA = simd_array<ResultElemT, N>(b);                                                                 \
            auto tmpB = simd_array<ResultElemT, N>(b);                                                                 \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        } else {                                                                                                       \
            auto tmpA = simd_array<ResultElemT, N>(b);                                                                 \
            auto &tmpB = b;                                                                                            \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        }                                                                                                              \
    }

AKR_SIMD_GEN_OPERATOR_(+, array_operator_add)
AKR_SIMD_GEN_OPERATOR_(-, array_operator_sub)
AKR_SIMD_GEN_OPERATOR_(*, array_operator_mul)
AKR_SIMD_GEN_OPERATOR_(/, array_operator_div)
AKR_SIMD_GEN_SCALAR_RIGHT_OPERATOR_(*, array_operator_scalar_mul)
AKR_SIMD_GEN_SCALAR_RIGHT_OPERATOR_(/, array_operator_scalar_div)
AKR_SIMD_GEN_SCALAR_LEFT_OPERATOR_(*, array_operator_scalar_mul)
AKR_SIMD_GEN_SCALAR_LEFT_OPERATOR_(/, array_operator_scalar_div)

template <typename T, size_t N>
auto select(const array_mask<N> &mask, const simd_array<T, N> &a, const simd_array<T, N> &b) {
    return array_select<T, N>::apply(mask, a, b);
}

int main() {
    simd_array<float, 32> a, b;
    a = a + b;
}
