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
#include <cstring>
#include <immintrin.h>
#include <type_traits>
#include <xmmintrin.h>

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
template <size_t N> struct array_padded_size<float, N> { constexpr static size_t value = N < 2 ? N : align_to<N, 4>; };
template <size_t N> struct array_padded_size<int32_t, N> {
    constexpr static size_t value = N < 2 ? N : align_to<N, 4>;
};
struct simdf32x8 {
    __m256 m;
};
static_assert(sizeof(simdf32x8) == sizeof(__m256));
struct simdf32x4 {
    __m128 m;
};
static_assert(sizeof(simdf32x4) == sizeof(__m128));
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

template <size_t N> struct array_mask_not {
    static array_mask<N> apply(const array_mask<N> &a) {
        array_mask<N> res;
        for (size_t i = 0; i < N; i++) {
            res[i] = !a[i];
        }
        return res;
    }
};

template <size_t N> struct array_mask_and {
    static array_mask<N> apply(const array_mask<N> &a, const array_mask<N> &b) {
        array_mask<N> res;
        for (size_t i = 0; i < N; i++) {
            res[i] = a[i] && b[i];
        }
        return res;
    }
};

template <size_t N> struct array_mask_or {
    static array_mask<N> apply(const array_mask<N> &a, const array_mask<N> &b) {
        array_mask<N> res;
        for (size_t i = 0; i < N; i++) {
            res[i] = a[i] || b[i];
        }
        return res;
    }
};

template <size_t N> struct array_mask : std::array<int32_t, array_padded_size<int32_t, N>::value> {
    array_mask(bool v = true) { std::memset(this, v, sizeof(*this)); }
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

//#define AKR_SIMD_GEN_VFLOAT_OPERATOR_(assign_op, delegate, intrin_sse, intrin_avx2)                                    \
//    template <size_t N> void delegate##_simd_float_impl(float *A, const float *B);                                     \
//    template <size_t N, size_t Shift> void delegate##_simd_float_impl_shift(float *A, const float *B) {                \
//        return delegate##_simd_float_impl<N - Shift>(A + Shift, B + Shift);                                            \
//    }                                                                                                                  \
//    template <size_t N> void delegate##_simd_float_impl(float *A, const float *B) {                                    \
//        if constexpr (N >= 8 && MaxISA >= AKR_ISA_AVX) {                                                               \
//            auto a = reinterpret_cast<__m256 *>(A);                                                                    \
//            auto b = reinterpret_cast<const __m256 *>(B);                                                              \
//            *a = _mm256_add_ps(*a, *b);                                                                                \
//            delegate##_simd_float_impl_shift<N, 8>(A, B);                                                              \
//        } else if constexpr (N >= 4 && MaxISA >= AKR_ISA_SSE) {                                                        \
//            auto a = reinterpret_cast<__m128 *>(A);                                                                    \
//            auto b = reinterpret_cast<const __m128 *>(B);                                                              \
//            *a = _mm_add_ps(*a, *b);                                                                                   \
//            delegate##_simd_float_impl_shift<N, 4>(A, B);                                                              \
//        } else {                                                                                                       \
//            for (int i = 0; i < N; i++) {                                                                              \
//                A[i] assign_op B[i];                                                                                   \
//            }                                                                                                          \
//        }                                                                                                              \
//    }                                                                                                                  \
//    template <size_t N> struct delegate<float, N> {                                                                    \
//        using A = simd_array<float, N>;                                                                                \
//        static void apply(A &a, const A &b) {                                                                          \
//            auto *A = reinterpret_cast<float *>(&a);                                                                   \
//            auto *B = reinterpret_cast<const float *>(&b);                                                             \
//            delegate##_simd_float_impl<A::padded_size>(A, B);                                                          \
//        }                                                                                                              \
//    };

#define AKR_SIMD_GEN_VFLOAT_OPERATOR_(assign_op, delegate, intrin_sse, intrin_avx2)                                    \
    template <size_t N> inline void delegate##_simd_float_impl(simd32_storage<N> &lhs, const simd32_storage<N> &rhs) { \
        using head = decltype(simd32_storage<N>::head);                                                                \
        constexpr size_t rest = simd32_storage<N>::n_rest;                                                             \
        if constexpr (std::is_same_v<head, simdf32x8>) {                                                               \
            lhs.head.m = intrin_avx2(lhs.head.m, rhs.head.m);                                                          \
        } else if constexpr (std::is_same_v<head, simdf32x4>) {                                                        \
            lhs.head.m = intrin_sse(lhs.head.m, rhs.head.m);                                                           \
        } else {                                                                                                       \
            for (int i = 0; i < N; i++) {                                                                              \
                lhs.head.m[i] assign_op rhs.head.m[i];                                                                 \
            }                                                                                                          \
        }                                                                                                              \
        if constexpr (rest > 0) {                                                                                      \
            delegate##_simd_float_impl(lhs.next, rhs.next);                                                            \
        }                                                                                                              \
    }                                                                                                                  \
    template <size_t N> struct delegate<float, N> {                                                                    \
        using A = simd_array<float, N>;                                                                                \
        static void apply(A &a, const A &b) { delegate##_simd_float_impl(a._m, b._m); }                                \
    };

AKR_SIMD_GEN_VFLOAT_OPERATOR_(+=, array_operator_add, _mm_add_ps, _mm256_add_ps)
AKR_SIMD_GEN_VFLOAT_OPERATOR_(-=, array_operator_sub, _mm_sub_ps, _mm256_sub_ps)
AKR_SIMD_GEN_VFLOAT_OPERATOR_(*=, array_operator_mul, _mm_mul_ps, _mm256_mul_ps)
AKR_SIMD_GEN_VFLOAT_OPERATOR_(/=, array_operator_div, _mm_div_ps, _mm256_div_ps)

template <typename T, size_t N>
auto select(const array_mask<N> &mask, const simd_array<T, N> &a, const simd_array<T, N> &b) {
    return array_select<T, N>::apply(mask, a, b);
}

int main() {
    simd_array<float, 32> a, b;
    for (int i = 0; i < 32; i++) {
        a[i] = 2 * i + 1;
        b[i] = 3 * i + 2;
    }
    a = a + b;
    for (int i = 0; i < 32; i++) {
        printf("%f\n", a[i]);
    }
}
