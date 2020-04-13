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
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <type_traits>
#include <xmmintrin.h>

namespace Akari {
#define AKR_INLINE __forceline
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
    template <typename T, size_t N> struct array_broadcast {
        static void apply(simd_array<T, N> &arr, const T &v) {
            for (size_t i = 0; i < N; i++) {
                arr[i] = v;
            }
        }
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

    AKR_SIMD_DEFAULT_OPERATOR(array_operator_add, +=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_sub, -=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_mul, *=)
    AKR_SIMD_DEFAULT_OPERATOR(array_operator_div, /=)
    AKR_SIMD_DEFAULT_SCALAR_OPERATOR(array_operator_scalar_mul, *=)
    AKR_SIMD_DEFAULT_SCALAR_OPERATOR(array_operator_scalar_div, /=)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_lt, <)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_le, <=)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_gt, >)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_ge, >=)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_eq, ==)
    AKR_SIMD_DEFAULT_CMP_OPERATOR(array_operator_neq, !=)

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

    AKR_SIMD_GEN_CMP_OPERATOR_(<, array_operator_lt)
    AKR_SIMD_GEN_CMP_OPERATOR_(<=, array_operator_le)
    AKR_SIMD_GEN_CMP_OPERATOR_(>, array_operator_gt)
    AKR_SIMD_GEN_CMP_OPERATOR_(>=, array_operator_ge)
    AKR_SIMD_GEN_CMP_OPERATOR_(==, array_operator_eq)
    AKR_SIMD_GEN_CMP_OPERATOR_(!=, array_operator_neq)

#define AKR_SIMD_GEN_VFLOAT_OPERATOR_(assign_op, delegate, intrin_sse, intrin_avx2)                                    \
    template <size_t N> struct delegate<float, N> {                                                                    \
        template <size_t M> inline static void simd_float_impl(simd32_storage<M> &lhs, const simd32_storage<M> &rhs) { \
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
        static void apply(A &a, const A &b) { simd_float_impl(a._m, b._m); }                                           \
    };
#define AKR_SIMD_GEN_VFLOAT_CMP_OPERATOR_(cmp_op, intrin_op, delegate, intrin_sse, intrin_avx2)                        \
    template <size_t N> struct delegate<float, N> {                                                                    \
        template <size_t M>                                                                                            \
        inline static void simd_float_impl(simd32_storage<M> &mask, const simd32_storage<M> &lhs,                      \
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
        static M apply(const A &a, const A &b) {                                                                       \
            M m;                                                                                                       \
            simd_float_impl(m._m, a._m, b._m);                                                                         \
            return m;                                                                                                  \
        }                                                                                                              \
    };
#define AKR_SIMD_GEN_VMASK_OPERATOR_(assign_op, delegate, intrin_sse, intrin_avx2)                                     \
    template <size_t N> struct delegate {                                                                              \
        template <size_t M> inline static void simd_float_impl(simd32_storage<M> &lhs, const simd32_storage<M> &rhs) { \
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
        static void apply(A &a, const A &b) { simd_float_impl(a._m, b._m); }                                           \
    };

    AKR_SIMD_GEN_VMASK_OPERATOR_(&=, array_mask_and, _mm_and_ps, _mm256_and_ps)
    AKR_SIMD_GEN_VMASK_OPERATOR_(|=, array_mask_or, _mm_or_ps, _mm256_or_ps)
    AKR_SIMD_GEN_VMASK_OPERATOR_(^=, array_mask_xor, _mm_xor_ps, _mm256_xor_ps)
    AKR_SIMD_GEN_MASK_OPERATOR_(&, array_mask_and)
    AKR_SIMD_GEN_MASK_OPERATOR_(|, array_mask_or)
    AKR_SIMD_GEN_MASK_OPERATOR_(^, array_mask_xor)
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

    template<typename T>
    struct _scalar_t{
        using type = T;
    };
    template<typename T,size_t N>
    struct _scalar_t<simd_array<T,N>>{
        using type = T;
    };
    template<typename T>
    using scalar_t = typename _scalar_t<T>::type;
    template <typename T, size_t N> struct simd_array : public array_storage<T, N> {
        simd_array() = default;
        simd_array(const T &v) { array_broadcast<T, N>::apply(*this, v); }
        T &operator[](size_t i) { return this->data[i]; }
        const T &operator[](size_t i) const { return this->data[i]; }
#define AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(assign_op, delegate)                                                       \
    simd_array &operator assign_op(const simd_array &rhs) {                                                            \
        delegate<T, N>::apply(*this, rhs);                                                                             \
        return *this;                                                                                                  \
    }
        AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(+=, array_operator_add)
        AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(-=, array_operator_sub)
        AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(*=, array_operator_mul)
        AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR(/=, array_operator_div)
#undef AKR_SIMD_GEN_VFLOAT_ASSIGN_OPERATOR
    };

    template <size_t N> struct array_mask : array_storage<int32_t, N> {
        array_mask(bool v = true) { std::memset(this, v ? 0xFF : 0, sizeof(*this)); }
        operator bool() const { return any(*this); }
        int32_t &operator[](size_t i) { return this->data[i]; }
        const int32_t &operator[](size_t i) const { return this->data[i]; }
#define AKR_SIMD_GEN_VMASK_ASSIGN_OPERATOR(assign_op, delegate)                                                        \
    array_mask &operator assign_op(const array_mask &rhs) {                                                            \
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

    template <size_t N> struct array_broadcast<float, N> {
        template <size_t M> static void apply(simd32_storage<M> &simd, const float &v) {
            using head = decltype(simd32_storage<M>::head);
            constexpr size_t rest = simd32_storage<M>::n_rest;
            if constexpr (std::is_same_v<head, simdf32x8>) {
                simd.head.m = _mm256_broadcast_ss(&v);
            } else if constexpr (std::is_same_v<head, simdf32x4>) {
                simd.head.m = _mm_broadcast_ss(&v);
            } else {
                for (int i = 0; i < N; i++) {
                    simd.head.m[i] = v;
                }
            }
            if constexpr (rest > 0) {
                apply(simd.next, v);
            }
        }
        static void apply(simd_array<float, N> &arr, const float &v) { apply(arr._m, v); }
    };

    template <size_t N> struct array_select<float, N> {
        using T = float;
        using A = simd_array<T, N>;
        template <size_t M>
        static void impl(simd32_storage<M> &res, const simd32_storage<M> &mask, const simd32_storage<M> &a,
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
        static A apply(const array_mask<N> &mask, const A &a, const A &b) {
            A tmp;
            impl<N>(tmp._m, mask._m, a._m, b._m);
            return tmp;
        }
    };

    template <typename T, size_t N>
    auto select(const array_mask<N> &mask, const simd_array<T, N> &a, const simd_array<T, N> &b) {
        return array_select<T, N>::apply(mask, a, b);
    }
    template <typename T, size_t N> using Array = simd_array<T, N>;

    template <size_t N> using Mask = array_mask<N>;
} // namespace Akari
#endif // AKARIRENDER_SIMDARRAYBASIC_HPP
