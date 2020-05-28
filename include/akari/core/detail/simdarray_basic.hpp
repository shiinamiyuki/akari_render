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

#ifndef AKARIRENDER_SIMDARRAYBASIC_HPP
#define AKARIRENDER_SIMDARRAYBASIC_HPP
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <type_traits>
#include <xmmintrin.h>

namespace akari {
#define AKR_INLINE    __forceline
#define AKR_ISA_NONE  0
#define AKR_ISA_SSE   1
#define AKR_ISA_SSE42 2
#define AKR_ISA_AVX   3
#define AKR_ISA_AVX2  4

    constexpr size_t MaxISA = AKR_ISA_AVX2;

    constexpr size_t ptr_size = sizeof(void *);
    template <size_t N, size_t Align = ptr_size> constexpr size_t align_to = ((N + Align - 1u) / Align) * Align;

    template <typename T, size_t N> struct array_alignment { constexpr static size_t value = align_to<sizeof(T)>; };
    template <typename T, size_t N> struct array_padded_size { constexpr static size_t value = N; };
    template <size_t N> struct array_padded_size<float, N> {
        constexpr static size_t value = N < 2 ? N : align_to<N, 4>;
    };
    template <size_t N> struct array_padded_size<int32_t, N> {
        constexpr static size_t value = N < 2 ? N : align_to<N, 4>;
    };
    struct simdf32x8 {
        union {
            __m256 m;
            __m256i mi;
        };
    };
    static_assert(sizeof(simdf32x8) == sizeof(__m256));
    struct simdf32x4 {
        union {
            __m128 m;
            __m128 mi;
        };
    };

    static_assert(sizeof(simdf32x4) == sizeof(__m128));
    using simdi32x4 = simdf32x4;
    using simdi32x8 = simdf32x8;

    template <size_t N> struct simd32_storage_type_list;

    template <size_t N, typename = void> struct simd32_storage {
        typename simd32_storage_type_list<N>::type head;
        constexpr static size_t n_rest = simd32_storage_type_list<N>::n_rest;
        simd32_storage<n_rest> next;
    };

    template <size_t N> struct simd32_storage_type_list {
        using type = std::conditional_t<N >= 8 && MaxISA >= AKR_ISA_AVX, simdf32x8,
                                        std::conditional_t<N >= 4 && MaxISA >= AKR_ISA_SSE, simdf32x4, float[N]>>;
        constexpr static size_t n_rest = N - sizeof(type) / sizeof(float);
    };

    template <size_t N>
    struct simd32_storage<N, std::void_t<std::enable_if_t<simd32_storage_type_list<N>::n_rest == 0, void>>> {
        typename simd32_storage_type_list<N>::type head;
        constexpr static size_t n_rest = 0;
    };

    template <typename T, size_t N> struct array_storage {
        constexpr static size_t padded_size = array_padded_size<T, N>::value;
        alignas(array_alignment<T, N>::value) T data[padded_size];
    };

    template <size_t N> struct array_storage<float, N> {
        using T = float;
        constexpr static size_t padded_size = array_padded_size<T, N>::value;
        union {
            alignas(array_alignment<T, N>::value) T data[padded_size];
            simd32_storage<padded_size> _m;
        };
    };
    template <size_t N> struct array_storage<int32_t, N> {
        using T = int32_t;
        constexpr static size_t padded_size = array_padded_size<float, N>::value;
        union {
            alignas(array_alignment<float, N>::value) int32_t data[padded_size];
            simd32_storage<padded_size> _m;
        };
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

    template <typename T, size_t N> struct simd_array;
    template <typename T> struct is_array : std::false_type {};
    template <typename T, size_t N> struct is_array<simd_array<T, N>> : std::true_type {};
    template <typename T> constexpr bool is_array_v = is_array<T>::value;
    template <typename T> struct is_scalar : std::true_type {};
    template <typename T, size_t N> struct is_scalar<simd_array<T, N>> : std::false_type {};
    template <size_t N> struct is_scalar<array_mask<N>> : std::false_type {};
    template <typename T> constexpr bool is_scalar_v = is_scalar<T>::value;
    template <typename T, size_t N> struct array_broadcast {
        static void apply(simd_array<T, N> &arr, const T &v) {
            for (size_t i = 0; i < N; i++) {
                arr[i] = v;
            }
        }
    };

    template <typename T, typename U, size_t N> struct array_from {
        static void apply(simd_array<T, N> &arr, const simd_array<U, N> &v) {
            for (size_t i = 0; i < N; i++) {
                arr[i] = T(v[i]);
            }
        }
    };

    // template <typename T, size_t N> struct masked_simd_array : simd_array<T, N> { array_mask<N> mask; };
#define AKR_SIMD_DEFAULT_FUNCTION_1(delegate, func)                                                                    \
    template <typename T, size_t N> struct delegate {                                                                  \
        using A = simd_array<T, N>;                                                                                    \
        static A apply(const A &x) {                                                                                   \
            A res;                                                                                                     \
            for (size_t i = 0; i < N; i++) {                                                                           \
                res[i] = func(x[i]);                                                                                   \
            }                                                                                                          \
            return res;                                                                                                \
        }                                                                                                              \
    };                                                                                                                 \
    template <typename T, size_t N> simd_array<T, N> func(const simd_array<T, N> &x) {                                 \
        return delegate<T, N>::apply(x);                                                                               \
    }
#define AKR_SIMD_DEFAULT_FUNCTION_2(delegate, func)                                                                    \
    template <typename T, size_t N> struct delegate {                                                                  \
        using A = simd_array<T, N>;                                                                                    \
        static A apply(const A &x, const A &y) {                                                                       \
            A res;                                                                                                     \
            for (size_t i = 0; i < N; i++) {                                                                           \
                res[i] = func(x[i], y[i]);                                                                             \
            }                                                                                                          \
            return res; /**/                                                                                           \
        }                                                                                                              \
    };                                                                                                                 \
    template <typename T, size_t N> simd_array<T, N> func(const simd_array<T, N> &x, const simd_array<T, N> &y) {      \
        return delegate<T, N>::apply(x, y);                                                                            \
    }

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
        static A apply(const array_mask<N> &mask, const A &a, const A &b) {
            A tmp;
            for (size_t i = 0; i < N; i++) {
                if (mask[i]) {
                    tmp[i] = a[i];
                } else {
                    tmp[i] = b[i];
                }
            }
            return tmp;
        }
    };
    template <size_t N> struct mask_select {
        using A = array_mask<N>;
        template <size_t M>
        static AKR_FORCEINLINE void impl(simd32_storage<M> &res, const simd32_storage<M> &mask, const simd32_storage<M> &a,
                         const simd32_storage<M> &b) {
            using head = decltype(simd32_storage<M>::head);
            constexpr size_t rest = simd32_storage<M>::n_rest;
            if constexpr (std::is_same_v<head, simdf32x8>) {
                res.head.m = _mm256_blendv_ps(b.head.m, a.head.m, mask.head.m);
            } else if constexpr (std::is_same_v<head, simdf32x4>) {
                res.head.m = _mm_blendv_ps(b.head.m, a.head.m, mask.head.m);
            } else {
                for (int i = 0; i < N; i++) {
                    res.head.m[i] = mask[i] ? a[i] : b[i];
                }
            }
            if constexpr (rest > 0) {
                impl(res.next, mask.next, a.next, b.next);
            }
        }
        static AKR_FORCEINLINE A apply(const array_mask<N> &mask, const A &a, const A &b) {
            A tmp;
            impl(tmp._m, mask._m, a._m, b._m);
            return tmp;
        }
    };
#define AKR_SIMD_DEFAULT_CMP_OPERATOR(delegate, op)                                                                    \
    template <typename T, size_t N> struct delegate {                                                                  \
        using A = simd_array<T, N>;                                                                                    \
        static array_mask<N> apply(const A &a, const A &b) {                                                           \
            array_mask<N> mask;                                                                                        \
            for (size_t i = 0; i < N; i++) {                                                                           \
                mask[i] = (a[i] op b[i]) ? 0xFFFFFFFF : 0;                                                             \
            }                                                                                                          \
            return mask;                                                                                               \
        }                                                                                                              \
    };

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
            auto tmpA = simd_array<ResultElemT, N>(a);                                                                 \
            auto &tmpB = b;                                                                                            \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        }                                                                                                              \
    }

#define AKR_SIMD_GEN_CMP_OPERATOR_(op, delegate)                                                                       \
    template <typename T, size_t N> inline auto operator op(const simd_array<T, N> &a, const simd_array<T, N> &b) {    \
        return delegate<T, N>::apply(a, b);                                                                            \
    }                                                                                                                  \
    template <typename T, size_t N> inline auto operator op(const simd_array<T, N> &a, const T &b) {                   \
        return delegate<T, N>::apply(a, simd_array<T, N>(b));                                                          \
    }                                                                                                                  \
    template <typename T, size_t N> inline auto operator op(const T &a, const simd_array<T, N> &b) {                   \
        return delegate<T, N>::apply(simd_array<T, N>(a), b);                                                          \
    }

#define AKR_SIMD_GEN_MASK_OPERATOR_(op, delegate)                                                                      \
    template <size_t N> inline auto operator op(const array_mask<N> &a, const array_mask<N> &b) {                      \
        auto tmp = a;                                                                                                  \
        delegate<N>::apply(tmp, b);                                                                                    \
        return tmp;                                                                                                    \
    }

#define AKR_SIMD_GEN_SCALAR_RIGHT_OPERATOR_(op, delegate)                                                              \
    template <typename T, typename U, size_t N, typename _ = std::enable_if_t<is_scalar_v<U>>>                         \
    inline auto operator op(const simd_array<T, N> &a, const U &b) {                                                   \
        using ResultElemT = decltype(std::declval<T>() + std::declval<U>());                                           \
        static_assert(std::is_same_v<T, ResultElemT> || std::is_same_v<U, ResultElemT>, "What ?");                     \
        if constexpr (std::is_same_v<T, ResultElemT>) {                                                                \
            auto tmpA = a;                                                                                             \
            auto tmpB = T(b);                                                                                          \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        } else {                                                                                                       \
            auto tmpA = simd_array<ResultElemT, N>(a);                                                                 \
            auto &tmpB = b;                                                                                            \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        }                                                                                                              \
    }

#define AKR_SIMD_GEN_SCALAR_LEFT_OPERATOR_(op, delegate)                                                               \
    template <typename T, typename U, size_t N, typename _ = std::enable_if_t<is_scalar_v<T>>>                         \
    inline auto operator op(const T &a, const simd_array<U, N> &b) {                                                   \
        using ResultElemT = decltype(std::declval<T>() + std::declval<U>());                                           \
        static_assert(std::is_same_v<T, ResultElemT> || std::is_same_v<U, ResultElemT>, "What ?");                     \
        if constexpr (std::is_same_v<T, ResultElemT>) {                                                                \
            auto tmpA = simd_array<ResultElemT, N>(a);                                                                 \
            auto tmpB = simd_array<ResultElemT, N>(b);                                                                 \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        } else {                                                                                                       \
            auto tmpA = simd_array<ResultElemT, N>(a);                                                                 \
            auto &tmpB = b;                                                                                            \
            delegate<T, N>::apply(tmpA, tmpB);                                                                         \
            return tmpA;                                                                                               \
        }                                                                                                              \
    }

#define AKR_SIMD_GEN_VFLOAT_OPERATOR_(assign_op, delegate, intrin_sse, intrin_avx2)                                    \
    template <size_t N> struct delegate<float, N> {                                                                    \
        template <size_t M> AKR_FORCEINLINE static void simd_float_impl(simd32_storage<M> &lhs, const simd32_storage<M> &rhs) { \
            using head = decltype(simd32_storage<M>::head);                                                            \
            constexpr size_t rest = simd32_storage<M>::n_rest;                                                         \
            if constexpr (std::is_same_v<head, simdf32x8>) {                                                           \
                lhs.head.m = intrin_avx2(lhs.head.m, rhs.head.m);                                                      \
            } else if constexpr (std::is_same_v<head, simdf32x4>) {                                                    \
                lhs.head.m = intrin_sse(lhs.head.m, rhs.head.m);                                                       \
            } else {                                                                                                   \
                for (int i = 0; i < N; i++) {                                                                          \
                    lhs.head.m[i] assign_op rhs.head.m[i];                                                             \
                }                                                                                                      \
            }                                                                                                          \
            if constexpr (rest > 0) {                                                                                  \
                simd_float_impl(lhs.next, rhs.next);                                                                   \
            }                                                                                                          \
        }                                                                                                              \
        using A = simd_array<float, N>;                                                                                \
        static AKR_FORCEINLINE void apply(A &a, const A &b) { simd_float_impl(a._m, b._m); }                                           \
    };
#define AKR_SIMD_GEN_VFLOAT_CMP_OPERATOR_(cmp_op, intrin_op, delegate, intrin_sse, intrin_avx2)                        \
    template <size_t N> struct delegate<float, N> {                                                                    \
        template <size_t M>                                                                                            \
        AKR_FORCEINLINE static void simd_float_impl(simd32_storage<M> &mask, const simd32_storage<M> &lhs,                      \
                                           const simd32_storage<M> &rhs) {                                             \
            using head = decltype(simd32_storage<M>::head);                                                            \
            constexpr size_t rest = simd32_storage<M>::n_rest;                                                         \
            if constexpr (std::is_same_v<head, simdf32x8>) {                                                           \
                mask.head.m = intrin_avx2(lhs.head.m, rhs.head.m, intrin_op);                                          \
            } else if constexpr (std::is_same_v<head, simdf32x4>) {                                                    \
                mask.head.m = intrin_sse(lhs.head.m, rhs.head.m, intrin_op);                                           \
            } else {                                                                                                   \
                for (int i = 0; i < N; i++) {                                                                          \
                    reinterpret_cast<int32_t *>(&mask.head.m[0])[i] =                                                  \
                        (lhs.head.m[i] cmp_op rhs.head.m[i]) ? 0xFFFFFFFF : 0;                                         \
                }                                                                                                      \
            }                                                                                                          \
            if constexpr (rest > 0) {                                                                                  \
                simd_float_impl(mask.next, lhs.next, rhs.next);                                                        \
            }                                                                                                          \
        }                                                                                                              \
        using A = simd_array<float, N>;                                                                                \
        using M = array_mask<N>;                                                                                       \
        static AKR_FORCEINLINE M apply(const A &a, const A &b) {                                                                       \
            M m;                                                                                                       \
            simd_float_impl(m._m, a._m, b._m);                                                                         \
            return m;                                                                                                  \
        }                                                                                                              \
    };
#define AKR_SIMD_GEN_VMASK_OPERATOR_(assign_op, delegate, intrin_sse, intrin_avx2)                                     \
    template <size_t N> struct delegate {                                                                              \
        template <size_t M> AKR_FORCEINLINE static void simd_float_impl(simd32_storage<M> &lhs, const simd32_storage<M> &rhs) { \
            using head = decltype(simd32_storage<M>::head);                                                            \
            constexpr size_t rest = simd32_storage<M>::n_rest;                                                         \
            if constexpr (std::is_same_v<head, simdf32x8>) {                                                           \
                lhs.head.m = intrin_avx2(lhs.head.m, rhs.head.m);                                                      \
            } else if constexpr (std::is_same_v<head, simdf32x4>) {                                                    \
                lhs.head.m = intrin_sse(lhs.head.m, rhs.head.m);                                                       \
            } else {                                                                                                   \
                for (int i = 0; i < N; i++) {                                                                          \
                    reinterpret_cast<uint32_t *>(lhs.head.m)[i] assign_op reinterpret_cast<const uint32_t *>(          \
                        rhs.head.m)[i];                                                                                \
                }                                                                                                      \
            }                                                                                                          \
            if constexpr (rest > 0) {                                                                                  \
                simd_float_impl(lhs.next, rhs.next);                                                                   \
            }                                                                                                          \
        }                                                                                                              \
        using A = array_mask<N>;                                                                                       \
        static AKR_FORCEINLINE void apply(A &a, const A &b) { simd_float_impl(a._m, b._m); }                                           \
    };

    AKR_SIMD_DEFAULT_OPERATOR(array_operator_add, +=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_sub, -=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_mul, *=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_div, /=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_or, |=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_and, &=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_xor, ^=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_lshift, <<=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_right, >>=)
    AKR_SIMD_DEFAULT_SCALAR_OPERATOR(array_operator_scalar_mul, *=)
    AKR_SIMD_DEFAULT_SCALAR_OPERATOR(array_operator_scalar_div, /=)
    AKR_SIMD_DEFAULT_SCALAR_OPERATOR(array_operator_scalar_add, +=)
    AKR_SIMD_DEFAULT_SCALAR_OPERATOR(array_operator_scalar_sub, -=)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_lt, <)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_le, <=)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_gt, >)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_ge, >=)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_eq, ==)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_neq, !=)

    using std::acos;
    using std::asin;
    using std::atan;
    using std::atan2;
    using std::cos;
    using std::pow;
    using std::sin;
    using std::sqrt;
    using std::tan;
    AKR_SIMD_DEFAULT_FUNCTION_1(array_floor, floor)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_ceil, ceil)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_sin, sin)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_cos, cos)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_tan, tan)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_acos, acos)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_asin, asin)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_atan, atan)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_exp, exp)
    AKR_SIMD_DEFAULT_FUNCTION_2(array_atan2, atan2)
    AKR_SIMD_DEFAULT_FUNCTION_1(array_sqrt, sqrt)
    AKR_SIMD_DEFAULT_FUNCTION_2(array_pow, pow)

    AKR_SIMD_GEN_OPERATOR_(+, array_operator_add)
    AKR_SIMD_GEN_OPERATOR_(-, array_operator_sub)
    AKR_SIMD_GEN_OPERATOR_(*, array_operator_mul)
    AKR_SIMD_GEN_OPERATOR_(/, array_operator_div)
    AKR_SIMD_GEN_OPERATOR_(|, array_operator_or)
    AKR_SIMD_GEN_OPERATOR_(&, array_operator_and)
    AKR_SIMD_GEN_OPERATOR_(^, array_operator_xor)
    AKR_SIMD_GEN_SCALAR_RIGHT_OPERATOR_(*, array_operator_scalar_mul)
    AKR_SIMD_GEN_SCALAR_RIGHT_OPERATOR_(/, array_operator_scalar_div)
    AKR_SIMD_GEN_SCALAR_LEFT_OPERATOR_(*, array_operator_mul)
    AKR_SIMD_GEN_SCALAR_LEFT_OPERATOR_(/, array_operator_div)
    AKR_SIMD_GEN_SCALAR_RIGHT_OPERATOR_(+, array_operator_add)
    AKR_SIMD_GEN_SCALAR_RIGHT_OPERATOR_(-, array_operator_sub)
    AKR_SIMD_GEN_SCALAR_LEFT_OPERATOR_(+, array_operator_add)
    AKR_SIMD_GEN_SCALAR_LEFT_OPERATOR_(-, array_operator_sub)

    AKR_SIMD_GEN_CMP_OPERATOR_(<, array_operator_lt)
    AKR_SIMD_GEN_CMP_OPERATOR_(<=, array_operator_le)
    AKR_SIMD_GEN_CMP_OPERATOR_(>, array_operator_gt)
    AKR_SIMD_GEN_CMP_OPERATOR_(>=, array_operator_ge)
    AKR_SIMD_GEN_CMP_OPERATOR_(==, array_operator_eq)
    AKR_SIMD_GEN_CMP_OPERATOR_(!=, array_operator_neq)

    AKR_SIMD_GEN_VMASK_OPERATOR_(&=, array_mask_and, _mm_and_ps, _mm256_and_ps)
    AKR_SIMD_GEN_VMASK_OPERATOR_(|=, array_mask_or, _mm_or_ps, _mm256_or_ps)
    AKR_SIMD_GEN_VMASK_OPERATOR_(^=, array_mask_xor, _mm_xor_ps, _mm256_xor_ps)
    AKR_SIMD_GEN_MASK_OPERATOR_(&, array_mask_and)
    AKR_SIMD_GEN_MASK_OPERATOR_(|, array_mask_or)
    AKR_SIMD_GEN_MASK_OPERATOR_(^, array_mask_xor)
    AKR_SIMD_GEN_MASK_OPERATOR_(&&, array_mask_and)
    AKR_SIMD_GEN_MASK_OPERATOR_(||, array_mask_or)
    AKR_SIMD_GEN_VFLOAT_CMP_OPERATOR_(<, _CMP_LT_OS, array_operator_lt, _mm_cmp_ps, _mm256_cmp_ps)
    AKR_SIMD_GEN_VFLOAT_CMP_OPERATOR_(<=, _CMP_LE_OS, array_operator_le, _mm_cmp_ps, _mm256_cmp_ps)
    AKR_SIMD_GEN_VFLOAT_CMP_OPERATOR_(>, _CMP_GT_OS, array_operator_gt, _mm_cmp_ps, _mm256_cmp_ps)
    AKR_SIMD_GEN_VFLOAT_CMP_OPERATOR_(>=, _CMP_GE_OS, array_operator_ge, _mm_cmp_ps, _mm256_cmp_ps)
    AKR_SIMD_GEN_VFLOAT_CMP_OPERATOR_(==, _CMP_EQ_OS, array_operator_eq, _mm_cmp_ps, _mm256_cmp_ps)
    AKR_SIMD_GEN_VFLOAT_CMP_OPERATOR_(!=, _CMP_NEQ_OS, array_operator_neq, _mm_cmp_ps, _mm256_cmp_ps)
    AKR_SIMD_GEN_VFLOAT_OPERATOR_(+=, array_operator_add, _mm_add_ps, _mm256_add_ps)
    AKR_SIMD_GEN_VFLOAT_OPERATOR_(-=, array_operator_sub, _mm_sub_ps, _mm256_sub_ps)
    AKR_SIMD_GEN_VFLOAT_OPERATOR_(*=, array_operator_mul, _mm_mul_ps, _mm256_mul_ps)
    AKR_SIMD_GEN_VFLOAT_OPERATOR_(/=, array_operator_div, _mm_div_ps, _mm256_div_ps)

    template <typename T> struct _scalar_t { using type = T; };
    template <typename T, size_t N> struct _scalar_t<simd_array<T, N>> { using type = T; };
    template <typename T> using scalar_t = typename _scalar_t<T>::type;

    template <typename T, size_t N> struct array_operator_arrow {
        static void *apply(const simd_array<T, N> &v);
        static void *apply(simd_array<T, N> &v);
    };
    template <typename T, size_t N, typename Derived> struct simd_array_base : public array_storage<T, N> {
        simd_array_base() = default;
        Derived &derived() { return static_cast<Derived &>(*this); }
        template <typename U> simd_array_base(const simd_array<U, N> &v) { array_from<T, U, N>::apply(derived(), v); }
        simd_array_base(const T &v) { array_broadcast<T, N>::apply(derived(), v); }
        T &operator[](size_t i) { return this->data[i]; }
        const T &operator[](size_t i) const { return this->data[i]; }
#define AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(assign_op, delegate)                                                       \
    AKR_FORCEINLINE Derived &operator assign_op(const Derived &rhs) {                                                                  \
        delegate<T, N>::apply(static_cast<Derived &>(*this), rhs);                                                     \
        return static_cast<Derived &>(*this);                                                                          \
    }
        AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(+=, array_operator_add)
        AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(-=, array_operator_sub)
        AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(*=, array_operator_mul)
        AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(/=, array_operator_div)
#undef AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR
#define AKR_SIMD_GEN_ASSCESS(Index, Name)                                                                              \
    const T &Name() const {                                                                                            \
        static_assert(N >= Index);                                                                                     \
        return (*this)[Index];                                                                                         \
    }                                                                                                                  \
    T &Name() {                                                                                                        \
        static_assert(N >= Index);                                                                                     \
        return (*this)[Index];                                                                                         \
    }
        AKR_SIMD_GEN_ASSCESS(0, x)
        AKR_SIMD_GEN_ASSCESS(1, y)
        AKR_SIMD_GEN_ASSCESS(2, z)
        AKR_SIMD_GEN_ASSCESS(3, w)
#undef AKR_SIMD_GEN_ASSCESS
        auto operator->() const { return array_operator_arrow<T, N>::apply(*this); }
        auto operator->() { return array_operator_arrow<T, N>::apply(*this); }
    };
    template <size_t N> struct array_mask_masked_assign {
        array_mask<N> &_array;
        array_mask<N> _mask;
        void operator=(const array_mask<N> &rhs) { _array = mask_select<N>::apply(_mask, rhs, _array); }
    };
    template <typename A, typename T, size_t N> struct simd_array_masked_assign {
        A &_array;
        array_mask<N> _mask;
        void operator=(const A &rhs) { _array = array_select<T, N>::apply(_mask, rhs, _array); }
    };
    template <typename T> struct simd_array_masked_assign<T, T, 1> {
        T &_array;
        bool _mask;
        void operator=(const T &rhs) { _array = _mask ? rhs : _array; }
    };

    template <typename T, size_t N> struct unique_instance_context {};

    template <typename T, size_t N, typename Derived, typename _ = void>
    struct simd_array_impl : simd_array_base<T, N, Derived> {
        using simd_array_base<T, N, Derived>::simd_array_base;
    };
    template <typename T, size_t N, typename Derived>
    struct simd_array_impl<T, N, Derived, std::enable_if_t<std::is_pointer_v<T>>> : simd_array_base<T, N, Derived> {
        using simd_array_base<T, N, Derived>::simd_array_base;
        unique_instance_context<T, N> _unique_ctx;
    };

    template <typename T, size_t N> struct simd_array : simd_array_impl<T, N, simd_array<T, N>> {
        using simd_array_impl<T, N, simd_array<T, N>>::simd_array_impl;
        using simd_array_impl<T, N, simd_array<T, N>>::operator[];
        auto operator[](array_mask<N> mask) { return masked(*this, mask); }
    };
    template <size_t N> struct array_mask;
    template <typename T, size_t N> struct unique_instance_context_base {
        static_assert(std::is_pointer_v<T>);
        simd_array<T, N> *_array = nullptr;
        std::array<T, N> _unique_instances;
        std::array<array_mask<N>, N> _active;
        int n_unique = 0;
        unique_instance_context_base() {
            for (auto &mask : _active) {
                mask.clear_all();
            }
        }
        void init(simd_array<T, N> *_arr) {
            if (_array != nullptr)
                return;
            _array = _arr;
            auto find_instance = [=](T ptr) -> int {
                for (int i = 0; i < n_unique; i++) {
                    if ((*_array)[i] == ptr) {
                        return i;
                    }
                }
                return -1;
            };
            for (int i = 0; i < N; i++) {
                auto inst = (*_array)[i];
                auto idx = find_instance(inst);
                if (idx == -1) {
                    _unique_instances[n_unique] = inst;
                    _active[n_unique].set(i);
                    n_unique++;
                } else {
                    _active[idx].set(i);
                }
            }
        }
    };

    template <size_t N> struct array_mask : array_storage<int32_t, N> {
        array_mask(bool v = true) { std::memset(this, v ? 0xFF : 0, sizeof(*this)); }
        int32_t &operator[](size_t i) { return this->data[i]; }
        const int32_t &operator[](size_t i) const { return this->data[i]; }
        void set(size_t i) { this->data[i] = 0xffffffff; }
        void clear(size_t i) { this->data[i] = 0; }
        void clear_all() { std::memset(this, 0, sizeof(*this)); }
#define AKR_SIMD_GEN_VMASK_ASSIGN_OPERATOR(assign_op, delegate)                                                        \
    AKR_FORCEINLINE array_mask &operator assign_op(const array_mask &rhs) {                                                            \
        delegate<N>::apply(*this, rhs);                                                                                \
        return *this;                                                                                                  \
    }
        AKR_SIMD_GEN_VMASK_ASSIGN_OPERATOR(&=, array_mask_and)
        AKR_SIMD_GEN_VMASK_ASSIGN_OPERATOR(|=, array_mask_or)
        AKR_SIMD_GEN_VMASK_ASSIGN_OPERATOR(^=, array_mask_xor)
#undef AKR_SIMD_GEN_VMASK_ASSIGN_OPERATOR
    };

    template <size_t N> struct array_mask_not {
        static array_mask<N> apply(const array_mask<N> &mask) {
            array_mask<N> res(1);
            return res ^ mask;
        }
    };
    template <size_t N> inline auto operator~(const array_mask<N> &mask) { return array_mask_not<N>::apply(mask); }
    template <size_t N> inline auto operator!(const array_mask<N> &mask) { return array_mask_not<N>::apply(mask); }
    template <size_t N> struct array_broadcast<float, N> {
        template <size_t M> static AKR_FORCEINLINE void apply(simd32_storage<M> &simd, const float &v) {
            using head = decltype(simd32_storage<M>::head);
            constexpr size_t rest = simd32_storage<M>::n_rest;
            if constexpr (std::is_same_v<head, simdf32x8>) {
                simd.head.m = _mm256_set1_ps(v);
            } else if constexpr (std::is_same_v<head, simdf32x4>) {
                simd.head.m = _mm_set1_ps(v);
            } else {
                for (int i = 0; i < N; i++) {
                    simd.head.m[i] = v;
                }
            }
            if constexpr (rest > 0) {
                apply(simd.next, v);
            }
        }
        static AKR_FORCEINLINE void apply(simd_array<float, N> &arr, const float &v) { apply(arr._m, v); }
    };
    template <size_t N> struct array_broadcast<int, N> {
        template <size_t M> static AKR_FORCEINLINE void apply(simd32_storage<M> &simd, const int &v) {
            using head = decltype(simd32_storage<M>::head);
            constexpr size_t rest = simd32_storage<M>::n_rest;
            if constexpr (std::is_same_v<head, simdf32x8>) {
                simd.head.m = _mm256_set1_ps(*(float *)&v);
            } else if constexpr (std::is_same_v<head, simdf32x4>) {
                simd.head.m = _mm_set1_ps(*(float *)&v);
            } else {
                for (int i = 0; i < N; i++) {
                    simd.head.m[i] = v;
                }
            }
            if constexpr (rest > 0) {
                apply(simd.next, v);
            }
        }
        static AKR_FORCEINLINE void apply(simd_array<int, N> &arr, const int &v) { apply(arr._m, v); }
    };
    template <size_t N> struct array_select<float, N> {
        using T = float;
        using A = simd_array<T, N>;
        template <size_t M>
        static AKR_FORCEINLINE void impl(simd32_storage<M> &res, const simd32_storage<M> &mask,
                                       const simd32_storage<M> &a, const simd32_storage<M> &b) {
            using head = decltype(simd32_storage<M>::head);
            constexpr size_t rest = simd32_storage<M>::n_rest;
            if constexpr (std::is_same_v<head, simdf32x8>) {
                res.head.m = _mm256_blendv_ps(b.head.m, a.head.m, mask.head.m);
            } else if constexpr (std::is_same_v<head, simdf32x4>) {
                res.head.m = _mm_blendv_ps(b.head.m, a.head.m, mask.head.m);
            } else {
                for (int i = 0; i < N; i++) {
                    res.head.m[i] = mask[i] ? a[i] : b[i];
                }
            }
            if constexpr (rest > 0) {
                impl(res.next, mask.next, a.next, b.next);
            }
        }
        static AKR_FORCEINLINE A apply(const array_mask<N> &mask, const A &a, const A &b) {
            A tmp;
            impl<N>(tmp._m, mask._m, a._m, b._m);
            return tmp;
        }
    };
    template <size_t N> struct array_select<int, N> {
        using T = float;
        using A = simd_array<T, N>;
        template <size_t M>
        static AKR_FORCEINLINE void impl(simd32_storage<M> &res, const simd32_storage<M> &mask,
                                       const simd32_storage<M> &a, const simd32_storage<M> &b) {
            using head = decltype(simd32_storage<M>::head);
            constexpr size_t rest = simd32_storage<M>::n_rest;
            if constexpr (std::is_same_v<head, simdf32x8>) {
                res.head.m = _mm256_blendv_ps(b.head.m, a.head.m, mask.head.m);
            } else if constexpr (std::is_same_v<head, simdf32x4>) {
                res.head.m = _mm_blendv_ps(b.head.m, a.head.m, mask.head.m);
            } else {
                for (int i = 0; i < N; i++) {
                    res.head.m[i] = mask[i] ? a[i] : b[i];
                }
            }
            if constexpr (rest > 0) {
                impl(res.next, mask.next, a.next, b.next);
            }
        }
        static AKR_FORCEINLINE A apply(const array_mask<N> &mask, const A &a, const A &b) {
            A tmp;
            impl<N>(tmp._m, mask._m, a._m, b._m);
            return tmp;
        }
    };
    template <typename T, size_t N>
    AKR_FORCEINLINE auto select(const array_mask<N> &mask, const simd_array<T, N> &a, const simd_array<T, N> &b) {
        return array_select<T, N>::apply(mask, a, b);
    }

    template <typename T, size_t N> using Array = simd_array<T, N>;

    template <size_t N> using Mask = array_mask<N>;

    template <typename T, size_t N> struct packed_t { using type = simd_array<T, N>; };
    template <typename T> struct packed_t<T, 1> { using type = T; };
    template <size_t N> struct packed_t<bool, N> { using type = array_mask<N>; };
    template <> struct packed_t<bool, 1> { using type = bool; };
    // pack N Ts into a scalar or array or mask
    template <typename T, size_t N> using Packed = typename packed_t<T, N>::type;

    template <typename T, typename = std::enable_if_t<is_scalar_v<T>>> auto select(bool m, const T &a, const T &b) {
        return m ? a : b;
    }
    template <typename T, size_t N>
    AKR_FORCEINLINE simd_array_masked_assign<simd_array<T, N>, T, N> masked(simd_array<T, N> &arr, const array_mask<N> &_mask) {
        return simd_array_masked_assign<simd_array<T, N>, T, N>{arr, _mask};
    }
    template <size_t N> AKR_FORCEINLINE array_mask_masked_assign<N> masked(array_mask<N> &arr, const array_mask<N> &_mask) {
        return array_mask_masked_assign<N>{arr, _mask};
    }
    template <typename T, typename _ = std::enable_if<is_scalar_v<T>>>
    AKR_FORCEINLINE simd_array_masked_assign<T, T, 1> masked(T &arr, const bool &_mask) {
        return simd_array_masked_assign<T, T, 1>{arr, _mask};
    }
    template <size_t N> struct simd_tag_t { constexpr static size_t lanes = N; };

    template <size_t N> static constexpr simd_tag_t<N> simd_tag{};

    AKR_FORCEINLINE bool any(bool x) { return x; }
    AKR_FORCEINLINE bool all(bool x) { return x; }
} // namespace akari
#endif // AKARIRENDER_SIMDARRAYBASIC_HPP
