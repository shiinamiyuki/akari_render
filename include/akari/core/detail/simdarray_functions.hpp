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

    template <size_t N> simd_array<float, N> floatbits(const simd_array<int, N> &x) {
        simd_array<float, N> y;
        static_assert(sizeof(x) == sizeof(y));
        std::memcpy(y.data, x.data, sizeof(x));
    }

    template <size_t N> simd_array<int, N> intbits(const simd_array<float, N> &x) {
        simd_array<int, N> y;
        static_assert(sizeof(x) == sizeof(y));
        std::memcpy(y.data, x.data, sizeof(x));
    }
    // from https://github.com/ispc/ispc/blob/master/stdlib.ispc
    template <typename F, typename I> inline F _sin(const F &x_full) {
        static const auto pi_over_two_vec = F(1.57079637050628662109375f);
        static const auto two_over_pi_vec = F(0.636619746685028076171875f);
        F scaled = x_full * two_over_pi_vec;
        F k_real = floor(scaled);
        I k = I(k_real);
        F x = x_full - k_real * pi_over_two_vec;
        I k_mod4 = I(k) & I(3);
        auto sin_usecos = ((k_mod4 == 1) | (k_mod4 == 3));
        auto flip_sign = (k_mod4 > 1);

        static const F sin_c2 = -0.16666667163372039794921875;
        static const F sin_c4 = 8.333347737789154052734375e-3;
        static const F sin_c6 = -1.9842604524455964565277099609375e-4;
        static const F sin_c8 = 2.760012648650445044040679931640625e-6;
        static const F sin_c10 = -2.50293279435709337121807038784027099609375e-8;

        static const F cos_c2 = -0.5;
        static const F cos_c4 = 4.166664183139801025390625e-2;
        static const F cos_c6 = -1.388833043165504932403564453125e-3;
        static const F cos_c8 = 2.47562347794882953166961669921875e-5;
        static const F cos_c10 = -2.59630184018533327616751194000244140625e-7;

        auto outside = select(sin_usecos, F(1.0f), x);
        auto c2 = select(sin_usecos, cos_c2, sin_c2);
        auto c4 = select(sin_usecos, cos_c4, sin_c4);
        auto c6 = select(sin_usecos, cos_c6, sin_c6);
        auto c8 = select(sin_usecos, cos_c8, sin_c8);
        auto c10 = select(sin_usecos, cos_c10, sin_c10);

        auto x2 = x * x;
        auto formula = x2 * c10 + c8;
        formula = x2 * formula + c6;
        formula = x2 * formula + c4;
        formula = x2 * formula + c2;
        formula = x2 * formula + F(1.0f);
        formula *= outside;
        formula = select(flip_sign, F(-1.0f) * formula, formula);
        return formula;
    }

    template <typename F, typename I> inline F _cos(const F &x_full) {
        static const auto pi_over_two_vec = F(1.57079637050628662109375f);
        static const auto two_over_pi_vec = F(0.636619746685028076171875f);
        F scaled = x_full * two_over_pi_vec;
        F k_real = floor(scaled);
        I k = I(k_real);
        F x = x_full - k_real * pi_over_two_vec;
        I k_mod4 = I(k) & I(3);
        auto cos_usecos = ((k_mod4 == 0) || (k_mod4 == 3));
        auto flip_sign = ((k_mod4 == 1) || (k_mod4 == 2));

        static const F sin_c2 = -0.16666667163372039794921875;
        static const F sin_c4 = 8.333347737789154052734375e-3;
        static const F sin_c6 = -1.9842604524455964565277099609375e-4;
        static const F sin_c8 = 2.760012648650445044040679931640625e-6;
        static const F sin_c10 = -2.50293279435709337121807038784027099609375e-8;

        static const F cos_c2 = -0.5;
        static const F cos_c4 = 4.166664183139801025390625e-2;
        static const F cos_c6 = -1.388833043165504932403564453125e-3;
        static const F cos_c8 = 2.47562347794882953166961669921875e-5;
        static const F cos_c10 = -2.59630184018533327616751194000244140625e-7;

        auto outside = select(cos_usecos, F(1.0f), x);
        auto c2 = select(cos_usecos, cos_c2, sin_c2);
        auto c4 = select(cos_usecos, cos_c4, sin_c4);
        auto c6 = select(cos_usecos, cos_c6, sin_c6);
        auto c8 = select(cos_usecos, cos_c8, sin_c8);
        auto c10 = select(cos_usecos, cos_c10, sin_c10);

        auto x2 = x * x;
        auto formula = x2 * c10 + c8;
        formula = x2 * formula + c6;
        formula = x2 * formula + c4;
        formula = x2 * formula + c2;
        formula = x2 * formula + F(1.0f);
        formula *= outside;
        formula = select(flip_sign, F(-1.0f) * formula, formula);
        return formula;
    }
    template <typename F, typename I> inline F ldexp(const F &x, const I &n) {
        I ex = 0x7F800000u;
        I ix = intbits(x);
        ex &= ix;               // extract old exponent;
        ix = ix & ~0x7F800000u; // clear exponent
        n = (n << 23) + ex;
        ix |= n; // insert new exponent
        return floatbits(ix);
    }

    template <typename F, typename I> inline F _fastexp(const F &x_full) {
        F z = floor(1.44269504088896341f * x_full + 0.5f);
        I n;
        x_full -= z * 0.693359375f;
        x_full -= z * -2.12194440e-4f;
        n = I(z);

        z = x_full * x_full;
        z = (((((1.9875691500E-4f * x_full + 1.3981999507E-3f) * x_full + 8.3334519073E-3f) * x_full +
               4.1665795894E-2f) *
                  x_full +
              1.6666665459E-1f) *
                 x_full +
             5.0000001201E-1f) *
                z +
            x_full + 1.f;
        x_full = ldexp(z, n);
        return x_full;
    }

    template <typename F, typename I> inline F _exp(const F &x_full) {
        const F ln2_part1 = 0.6931457519;
        const F ln2_part2 = 1.4286067653e-6;
        const F one_over_ln2 = 1.44269502162933349609375;

        const F scaled = x_full * one_over_ln2;
        const F k_real = floor(scaled);
        const I k = I(k_real);

        // Reduced range version of x
        F x = x_full - k_real * ln2_part1;
        x -= k_real * ln2_part2;

        // These coefficients are for e^x in [0, ln(2)]
        const F one = 1.;
        const F c2 = 0.4999999105930328369140625;
        const F c3 = 0.166668415069580078125;
        const F c4 = 4.16539050638675689697265625e-2;
        const F c5 = 8.378830738365650177001953125e-3;
        const F c6 = 1.304379315115511417388916015625e-3;
        const F c7 = 2.7555381529964506626129150390625e-4;

        F result = x * c7 + c6;
        result = x * result + c5;
        result = x * result + c4;
        result = x * result + c3;
        result = x * result + c2;
        result = x * result + one;
        result = x * result + one;

        // Compute 2^k (should differ for uniform float and double, but I'll avoid
        // it for now and just do uniform floats)
        const F fpbias = 127;
        I biased_n = k + fpbias;
        auto overflow = k > fpbias;
        // Minimum exponent is -126, so if k is <= -127 (k + 127 <= 0)
        // we've got underflow. -127 * ln(2) -> -88.02. So the most
        // negative uniform float input that doesn't result in zero is like -88.
        auto underflow = (biased_n <= 0);
        const I InfBits = 0x7f800000;
        biased_n <<= 23;
        // Reuniform interpret this thing as uniform float
        F two_to_the_n = floatbits(biased_n);
        // Handle both doubles and uniform floats (hopefully eliding the copy for uniform float)
        F elemtype_2n = two_to_the_n;
        result *= elemtype_2n;
        result = select(overflow, floatbits(InfBits), result);
        result = select(underflow, 0.0, result);
        return result;
    }

    template <size_t N> struct array_sin<float, N> {
        using A = simd_array<float, N>;
        static A apply(const A &x) { return _sin<simd_array<float, N>, simd_array<int, N>>(x); }
    };
    template <size_t N> struct array_cos<float, N> {
        using A = simd_array<float, N>;
        static A apply(const A &x) { return _cos<simd_array<float, N>, simd_array<int, N>>(x); }
    };
    template <size_t N> struct array_exp<float, N> {
        using A = simd_array<float, N>;
        static A apply(const A &x) { return _exp<simd_array<float, N>, simd_array<int, N>>(x); }
    };

    template <size_t N> simd_array<float, N> fastexp(const simd_array<float, N> &x) {
        return _fastexp<simd_array<float, N>, simd_array<int, N>>(x);
    }



#endif
} // namespace akari
#endif // AKARIRENDER_SIMDARRAYFUNCTIONS_HPP
