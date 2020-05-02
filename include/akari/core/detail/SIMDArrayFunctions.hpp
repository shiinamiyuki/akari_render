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

#ifndef AKARIRENDER_SIMDARRAYFUNCTIONS_HPP
#define AKARIRENDER_SIMDARRAYFUNCTIONS_HPP

namespace akari {
#ifdef __INTEL_COMPILER
#define AKR_GEN_SIMD_VFLOAT_FUNCTION(func, intrin_sse, intrin_avx)                                                     \
    template <size_t N> struct simd_float_##func {                                                                     \
        template <size_t M> static void impl(simd32_storage<M> &a, const simd32_storage<M> &b) {                       \
            using ::std::func;                                                                                         \
            using head = decltype(simd32_storage<M>::head);                                                            \
            constexpr size_t rest = simd32_storage<M>::n_rest;                                                         \
            if constexpr (std::is_same_v<head, simdf32x8>) {                                                           \
                a.head.m = intrin_avx(b.head.m);                                                                       \
            } else if constexpr (std::is_same_v<head, simdf32x4>) {                                                    \
                a.head.m = intrin_sse(b.head.m);                                                                       \
            } else {                                                                                                   \
                for (int i = 0; i < N; i++) {                                                                          \
                    a.head.m[i] = func(b.jead[i]);                                                                     \
                }                                                                                                      \
            }                                                                                                          \
            if constexpr (rest > 0) {                                                                                  \
                impl(a.next, b.next);                                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    };                                                                                                                 \
    template <size_t N> simd_array<float, N> func(const simd_array<float, N> &x) {                                     \
        simd_array<float, N> res;                                                                                      \
        simd_float_##func<N>::impl(res._m, x._m);                                                                      \
    }
    AKR_GEN_SIMD_VFLOAT_FUNCTION(sin, _mm_sin_ps, _mm256_sin_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(cos, _mm_cos_ps, _mm256_cos_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(tan, _mm_tan_ps, _mm256_tan_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(sinh, _mm_sinh_ps, _mm256_sinh_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(cosh, _mm_cosh_ps, _mm256_cosh_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(tanh, _mm_tanh_ps, _mm256_tanh_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(asin, _mm_asin_ps, _mm256_asin_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(acos, _mm_acos_ps, _mm256_acos_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(atan, _mm_atan_ps, _mm256_atan_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(asinh, _mm_asinh_ps, _mm256_asinh_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(acosh, _mm_acosh_ps, _mm256_acosh_ps)
    AKR_GEN_SIMD_VFLOAT_FUNCTION(atanh, _mm_atanh_ps, _mm256_atanh_ps)
#undef AKR_GEN_SIMD_VFLOAT_FUNCTION
#else
    // we need to write our own math functions
    template <size_t N> struct simd_float_fma {
        template <size_t M>
        static void impl(const simd32_storage<M> &res, const simd32_storage<M> &a, const simd32_storage<M> &b,
                         const simd32_storage<M> &c) {
            using ::std::fma;
            using head = decltype(simd32_storage<M>::head);
            constexpr size_t rest = simd32_storage<M>::n_rest;
            if constexpr (std::is_same_v<head, simdf32x8>) {
                res.head.m = _mm256_fmadd_ps(a.head.m, b.head.m, c.head.m);
            } else if constexpr (std::is_same_v<head, simdf32x4>) {
                res.head.m = _mm_fmadd_ps(a.head.m, b.head.m, c.head.m);
            } else {
                for (int i = 0; i < N; i++) {
                    res.head.m[i] = fma(a.head[i], b.head[i], c.head[i]);
                }
            }
            if constexpr (rest > 0) {
                impl(a.next, b.next);
            }
        }
    };

    // (x * y) + z
    template <size_t N>
    inline simd_array<float, N> fma(const simd_array<float, N> &x, const simd_array<float, N> &y,
                                    const simd_array<float, N> &z) {
        simd_array<float, N> r;
        simd_float_fma<N>::impl(r._m, x._m, y._m, z._m);
        return r;
    }
    inline float fma(float x, float y, float z) { return std::fma(x, y, z); }

    template <typename T, typename S = scalar_t<T>> inline T poly2(const T &x, const S &c0, const S &c1, const S &c2) {
        auto x2 = x * x;
        return fma(x2, T(c2), fma(x, T(c1), T(c0)));
    }

    template <typename T, typename S = scalar_t<T>>
    inline T poly3(const T &x, const S &c0, const S &c1, const S &c2, const S &c3) {
        auto x2 = x * x;
        return fma(x2, fma(x, T(c3), T(c2)), fma(x, T(c1), T(c0)));
    }

    template <typename T, typename S = scalar_t<T>>
    inline T poly4(const T &x, const S &c0, const S &c1, const S &c2, const S &c3, const S &c4) {
        auto x2 = x * x;
        auto x4 = x2 * x2;
        return fma(x2, fma(x, T(c3), T(c2)), fma(x, T(c1), T(c0))) + x4 * c4;
    }

    template <typename T, typename S = scalar_t<T>>
    inline T poly5(const T &x, const S &c0, const S &c1, const S &c2, const S &c3, const S &c4, const S &c5) {
        auto x2 = x * x;
        auto x4 = x2 * x2;
        return fma(x4, fma(x, T(c5), T(c4)), fma(x2, fma(x, T(c3), T(c2)), fma(x, T(c1), T(c0))));
    }

    template <typename T, typename S = scalar_t<T>>
    inline T poly6(const T &x, const S &c0, const S &c1, const S &c2, const S &c3, const S &c4, const S &c5,
                   const S &c6) {
        auto x2 = x * x;
        auto x4 = x2 * x2;
        return x4 * (fma(x, T(c5), T(c4)) + x2 * c6) + fma(x2, fma(x, T(c3), T(c2)), fma(x, T(c1), T(c0)));
    }
#endif
} // namespace akari
#endif // AKARIRENDER_SIMDARRAYFUNCTIONS_HPP
