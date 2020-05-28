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

#include <akari/core/platform.h>
#include <immintrin.h>
#include <type_traits>
#include <array>
#include <cstring>
namespace akari {
    AKR_FORCEINLINE float uintBitsToFloat(uint32_t x) {
        float v;
        std::memcpy(&v, &x, sizeof(float));
        return v;
    }
    AKR_FORCEINLINE float intBitsToFloat(int32_t x) {
        float v;
        std::memcpy(&v, &x, sizeof(float));
        return v;
    }
    AKR_FORCEINLINE uint32_t floatBitsToInt(float x) {
        uint32_t v;
        std::memcpy(&v, &x, sizeof(float));
        return v;
    }
    AKR_FORCEINLINE int32_t floatBitsToUint(float x) {
        int32_t v;
        std::memcpy(&v, &x, sizeof(float));
        return v;
    }
    template <typename T, size_t N, typename Derived> struct ArrayOperatorDefault {
#define AKR_ARRAY_DEFAULT_OP(op, func)                                                                                 \
    AKR_FORCEINLINE Derived func(const Derived &rhs) const {                                                             \
        Derived tmp;                                                                                                   \
        for (size_t i = 0; i < N; i++) {                                                                               \
            tmp.s[i] = (derived()->s[i] op rhs.s[i]);                                                                  \
        }                                                                                                              \
        return tmp;                                                                                                    \
    }
        AKR_ARRAY_DEFAULT_OP(+, add_)
        AKR_ARRAY_DEFAULT_OP(-, sub_)
        AKR_ARRAY_DEFAULT_OP(*, mul_)
        AKR_ARRAY_DEFAULT_OP(/, div_)
        AKR_ARRAY_DEFAULT_OP(&, and_)
        AKR_ARRAY_DEFAULT_OP(|, or_)
        AKR_ARRAY_DEFAULT_OP(^, xor_)
#undef AKR_ARRAY_DEFAULT_OP
      private:
        Derived *derived() { return static_cast<Derived *>(this); }
        const Derived *derived() const { return static_cast<const Derived *>(this); }
    };
    template <typename T, size_t N> struct ArrayStorage : ArrayOperatorDefault<T, N, ArrayStorage<T, N>> {
        using element_type = T;
        using mask_t = ArrayStorage<int32_t, N>;
        using Self = ArrayStorage;
        static constexpr bool is_specialized = false;
        std::array<T, N> s;
        template <typename F> AKR_FORCEINLINE ArrayStorage apply(F &&f) const {
            ArrayStorage tmp;
            for (size_t i = 0; i < N; i++) {
                tmp.s[i] = f(s[i]);
            }
            return tmp;
        }
        ArrayStorage() = default;
        ArrayStorage(const T &x) {
            for (auto &i : s) {
                i = x;
            }
        }
    };
    template <typename Derived> struct SIMD32x4 : ArrayOperatorDefault<float, 4, Derived> {
        using element_type = float;
        static constexpr bool is_specialized = true;
        using Self = SIMD32x4;
        union {
            __m128 f32x4;
            __m128i i32x4;
            float s[4];
        };
        SIMD32x4() = default;
        SIMD32x4(float x) : f32x4(_mm_set1_ps(x)) {}
        SIMD32x4(__m128 x) : f32x4(x) {}
#define AKR_ARRAY_OP(func, intrin)                                                                                     \
    AKR_FORCEINLINE Derived func(const Derived &rhs) const { return Derived(intrin(this->f32x4, rhs.f32x4)); }
        AKR_ARRAY_OP(add_, _mm_add_ps)
        AKR_ARRAY_OP(sub_, _mm_sub_ps)
        AKR_ARRAY_OP(mul_, _mm_mul_ps)
        AKR_ARRAY_OP(div_, _mm_div_ps)
        AKR_ARRAY_OP(min_, _mm_min_ps)
        AKR_ARRAY_OP(max_, _mm_max_ps)
        AKR_ARRAY_OP(and_, _mm_and_ps)
        AKR_ARRAY_OP(or_, _mm_or_ps)
        AKR_ARRAY_OP(xor_, _mm_xor_ps)
#undef AKR_ARRAY_OP
    };
    template <typename Derived> struct SIMDi32x4 : ArrayOperatorDefault<int, 4, Derived> {
        using element_type = float;
        static constexpr bool is_specialized = true;
        using Self = SIMDi32x4;
        union {
            __m128 f32x4;
            __m128i i32x4;
            float s[4];
        };
        SIMDi32x4() = default;
        SIMDi32x4(int32_t x) : i32x4(_mm_set1_epi32(x)) {}
        SIMDi32x4(__m128i x) : i32x4(x) {}
#define AKR_ARRAY_OP(func, intrin)                                                                                     \
    AKR_FORCEINLINE Derived func(const Derived &rhs) const { return Derived(intrin(this->f32x4, rhs.f32x4)); }
        AKR_ARRAY_OP(and_, _mm_and_ps)
        AKR_ARRAY_OP(or_, _mm_or_ps)
        AKR_ARRAY_OP(xor_, _mm_xor_ps)
#undef AKR_ARRAY_OP
    };
    template <typename Derived> struct SIMD32x8 : ArrayOperatorDefault<float, 8, Derived> {
        using element_type = float;
        using Self = SIMD32x8;
        static constexpr bool is_specialized = true;
        union {
            __m256 f32x8;
            __m256i i32x8;
            float s[8];
        };
        SIMD32x8() = default;
        SIMD32x8(float x) : f32x8(_mm256_set1_ps(x)) {}
        SIMD32x8(__m256 x) : f32x8(x) {}
#define AKR_ARRAY_OP(func, intrin)                                                                                     \
    AKR_FORCEINLINE Derived func(const Derived &rhs) const { return Derived(intrin(this->f32x8, rhs.f32x8)); }
        AKR_ARRAY_OP(add_, _mm256_add_ps)
        AKR_ARRAY_OP(sub_, _mm256_sub_ps)
        AKR_ARRAY_OP(mul_, _mm256_mul_ps)
        AKR_ARRAY_OP(div_, _mm256_div_ps)
        AKR_ARRAY_OP(min_, _mm256_min_ps)
        AKR_ARRAY_OP(max_, _mm256_max_ps)
        AKR_ARRAY_OP(and_, _mm256_and_ps)
        AKR_ARRAY_OP(or_, _mm256_or_ps)
        AKR_ARRAY_OP(xor_, _mm256_xor_ps)
#undef AKR_ARRAY_OP
    };
    template <typename Derived> struct SIMDi32x8 : ArrayOperatorDefault<int, 8, Derived> {
        using element_type = float;
        using Self = SIMDi32x8;
        union {
            __m256 f32x8;
            __m256i i32x8;
            float s[8];
        };
        SIMDi32x8() = default;
        SIMDi32x8(int32_t x) : f32x8(_mm256_set1_epi32(x)) {}
        SIMDi32x8(__m256i x) : i32x8(x) {}
#define AKR_ARRAY_OP(func, intrin)                                                                                     \
    AKR_FORCEINLINE Derived func(const Derived &rhs) const { return Derived(intrin(this->f32x8, rhs.f32x8)); }
        AKR_ARRAY_OP(and_, _mm256_and_ps)
        AKR_ARRAY_OP(or_, _mm256_or_ps)
        AKR_ARRAY_OP(xor_, _mm256_xor_ps)
#undef AKR_ARRAY_OP
    };
#define AKR_SPECIALIZE(Ty, N, Base)                                                                                    \
    template <> struct ArrayStorage<Ty, N> : Base<ArrayStorage<Ty, N>> {                                               \
        using element_type = Ty;                                                                                       \
        static constexpr bool is_specialized = true;                                                                   \
        using Self = ArrayStorage;                                                                                     \
        using Base<Self>::Base;                                                                                        \
    };

    AKR_SPECIALIZE(float, 4, SIMD32x4)
    AKR_SPECIALIZE(uint32_t, 4, SIMDi32x4)
    AKR_SPECIALIZE(int32_t, 4, SIMDi32x4)
    AKR_SPECIALIZE(float, 8, SIMD32x8)
    AKR_SPECIALIZE(uint32_t, 8, SIMDi32x8)
    AKR_SPECIALIZE(int32_t, 8, SIMDi32x8)
    constexpr size_t minmaxpow2(size_t x) {
        size_t i = 1;
        while (i != x) {
            auto t = i;
            i = i << 1;
            if (i >= x) {
                return t;
            }
        }
        return i;
    }
    constexpr size_t adjusted_array_size(size_t x) {
        if (x <= 2) {
            return x;
        }
        x = 4 * ((x + 3) / 4);
        return x;
    }
    template <typename T, size_t N, size_t _M = adjusted_array_size(N)> struct Array;
    template <typename T, size_t N, size_t M = minmaxpow2(N)> struct ArrayBaseImpl;
    template <typename T, size_t N, typename Derived> struct ArrayBase;
    template <typename Array1, typename Array2> struct ArrayStoragePair {
        // using Array1 = ArrayStorage<T, N1>;
        // using Array2 = ArrayBaseImpl<T, N2>;
        using T = typename Array1::element_type;
        static_assert(std::is_same_v<T, typename Array2::element_type>);
        Array1 lo;
        Array2 hi;
        using Self = ArrayStoragePair;
        ArrayStoragePair() = default;
        ArrayStoragePair(const T &a) : lo(a), hi(a) {}
        ArrayStoragePair(const Array1 &a, const Array2 &b) : lo(a), hi(b) {}
#define AKR_ARRAY_OP(func)                                                                                             \
    AKR_FORCEINLINE Self func(const Self &rhs) const { return Self(lo.func(rhs.lo), hi.func(rhs.hi)); }
        AKR_ARRAY_OP(add_)
        AKR_ARRAY_OP(sub_)
        AKR_ARRAY_OP(mul_)
        AKR_ARRAY_OP(div_)
        AKR_ARRAY_OP(min_)
        AKR_ARRAY_OP(max_)
        AKR_ARRAY_OP(and_)
        AKR_ARRAY_OP(or_)
        AKR_ARRAY_OP(xor_)
#undef AKR_ARRAY_OP
    };
    template <typename T, size_t N, size_t M> struct ArrayStoragePair_Indirect {
        using type = ArrayStoragePair<ArrayStorage<T, N>, ArrayStorage<T, M>>;
    };
    template <typename T, size_t N> struct ArrayStorage_Indirect { using type = ArrayStorage<T, N>; };
    template <typename T, size_t N, size_t M>
    using ArrayBaseImplBase = typename std::conditional_t<
        ArrayStorage<T, N>::is_specialized, ArrayStorage_Indirect<T, N>,
        std::conditional_t<N == M, ArrayStorage_Indirect<T, N>, ArrayStoragePair_Indirect<T, M, N - M>>>::type;
    template <typename T, size_t N, size_t M> struct ArrayBaseImpl : ArrayBaseImplBase<T, N, M> {
        using Self = ArrayBaseImpl;
        using Base = ArrayBaseImplBase<T, N, M>;
        ArrayBaseImpl() = default;
        ArrayBaseImpl(const T &v) : Base(v) {}
        ArrayBaseImpl(const Base &rhs) : Base(rhs) {}
#define AKR_ARRAY_OP(func)                                                                                             \
    AKR_FORCEINLINE Self func(const Self &rhs) const { return Self(static_cast<const Base *>(this)->func(rhs)); }
        AKR_ARRAY_OP(add_)
        AKR_ARRAY_OP(sub_)
        AKR_ARRAY_OP(mul_)
        AKR_ARRAY_OP(div_)
        AKR_ARRAY_OP(min_)
        AKR_ARRAY_OP(max_)
        AKR_ARRAY_OP(and_)
        AKR_ARRAY_OP(or_)
        AKR_ARRAY_OP(xor_)
#undef AKR_ARRAY_OP
    };
    template <typename T, size_t N, typename Derived> struct ArrayBaseCommon : ArrayBaseImpl<T, N> {
        using Self = ArrayBaseCommon;
        using Base = ArrayBaseImpl<T, N>;
        using Base::ArrayBaseImpl;
        ArrayBaseCommon(const Base &rhs) : Base(rhs) {}
#define AKR_ARRAY_OP(func)                                                                                             \
    AKR_FORCEINLINE Derived func(const Self &rhs) const { return Self(static_cast<const Base *>(this)->func(rhs)); }
        AKR_ARRAY_OP(add_)
        AKR_ARRAY_OP(sub_)
        AKR_ARRAY_OP(mul_)
        AKR_ARRAY_OP(div_)
        AKR_ARRAY_OP(min_)
        AKR_ARRAY_OP(max_)
        AKR_ARRAY_OP(and_)
        AKR_ARRAY_OP(or_)
        AKR_ARRAY_OP(xor_)
#undef AKR_ARRAY_OP
#define AKR_ARRAY_IOP(op, func)                                                                                        \
    AKR_FORCEINLINE Derived &operator op(const Self &rhs) {                                                              \
        *this = Derived(static_cast<const Base *>(this)->func(rhs));                                                   \
        return *this;                                                                                                  \
    }
        AKR_ARRAY_IOP(+=, add_)
        AKR_ARRAY_IOP(-=, sub_)
        AKR_ARRAY_IOP(*=, mul_)
        AKR_ARRAY_IOP(/=, div_)
#undef AKR_ARRAY_IOP
    };
    template <typename T, size_t N, typename Derived> struct ArrayBase : ArrayBaseCommon<T, N, Derived> {
        using Base = ArrayBaseCommon<T, N, Derived>;
        using Base::ArrayBaseCommon;
        AKR_FORCEINLINE T *raw() { return reinterpret_cast<T *>(this); }
        AKR_FORCEINLINE const T *raw() const { return reinterpret_cast<T *>(this); }
        AKR_FORCEINLINE T &operator[](size_t idx) { return raw()[idx]; }
        AKR_FORCEINLINE const T &operator[](size_t idx) const { return raw()[idx]; }
        template <size_t Idx> T &at() {
            static_assert(Idx < N);
            return raw()[Idx];
        }
        template <size_t Idx> const T &at() const {
            static_assert(Idx < N);
            return raw()[Idx];
        }
    };
    template <typename T, size_t N, size_t _M> struct Array : ArrayBase<T, _M, Array<T, N, _M>> {
        using Base = ArrayBase<T, _M, Array<T, N, _M>>;
        using Base::ArrayBase;
        template <typename... Args,
                  typename = std::enable_if_t<(sizeof...(Args) > 1) && std::conjunction_v<std::is_same<Args, T>...>>>
        Array(Args &&... args) {
            static_assert(sizeof...(args) <= N);
            T _tmp[] = {args...};
            std::memcpy(&this->template at<0>(), &_tmp[0], sizeof(_tmp));
            // if constexpr ()
        }
#define AKR_ARRAY_ACCESS_OP(name, idx)                                                                                 \
    const T &name() const { return this->template at<idx>(); }                                                                        \
    T &name() { return this->template at<idx>(); }
        AKR_ARRAY_ACCESS_OP(x, 0)
        AKR_ARRAY_ACCESS_OP(y, 1)
        AKR_ARRAY_ACCESS_OP(z, 2)
        AKR_ARRAY_ACCESS_OP(w, 2)
#undef AKR_ARRAY_ACCESS_OP
    };
    template <size_t N> using Mask = Array<uint32_t, N>;
    template <typename T> struct is_array : std::false_type {};
    template <typename T, size_t N> struct is_array<Array<T, N>> : std::true_type {};
    template <typename T> constexpr bool is_array_v = is_array<T>::value;
    template <typename T> struct array_size_ { static constexpr size_t value = 1; };
    template <typename T, size_t N> struct array_size_<Array<T, N>> { static constexpr size_t value = N; };
    template <typename T> constexpr size_t array_size_v = array_size_<T>::value;
    template <typename T> struct is_scalar : std::true_type {};
    template <typename T, size_t N> struct is_scalar<Array<T, N>> : std::false_type {};
    template <size_t N> struct is_scalar<Mask<N>> : std::false_type {};
    template <typename T> constexpr bool is_scalar_v = is_scalar<T>::value;
    template <typename T> struct scalar_ { using type = T; };
    template <typename T, size_t N> struct scalar_<Array<T, N>> { using type = T; };
    template <typename T> using scalar_t = typename scalar_<T>::type;
    template <typename T, typename U> struct replace_scalar_ {};
    template <typename T, size_t N, typename U> struct replace_scalar_<Array<T, N>, U> { using type = Array<U, N>; };
    template <typename T> using scalar_t = typename scalar_<T>::type;
    template <typename T, typename U> using replace_scalar_t = typename replace_scalar_<T, U>::type;

#define AKR_ARRAY_OP(op, func)                                                                                         \
    template <typename T, typename U, typename = std::enable_if_t<is_array_v<T> || is_array_v<U>>>                     \
    inline auto operator op(const T &a, const U &b) {                                                                  \
        using ST = scalar_t<T>;                                                                                        \
        using SU = scalar_t<U>;                                                                                        \
        constexpr auto N = is_array_v<T> ? array_size_v<T> : array_size_v<U>;                                          \
        using R = decltype(std::declval<ST>() + std::declval<SU>());                                                   \
        if constexpr (!is_array_v<T>) {                                                                                \
            auto arr_a = Array<R, N>(a);                                                                               \
            return a.func(b);                                                                                          \
        } else if constexpr (!is_array_v<U>) {                                                                         \
            auto arr_b = Array<R, N>(b);                                                                               \
            return a.func(arr_b);                                                                                      \
        } else {                                                                                                       \
            return a.func(b);                                                                                          \
        }                                                                                                              \
    }
    AKR_ARRAY_OP(+, add_)
    AKR_ARRAY_OP(-, sub_)
    AKR_ARRAY_OP(*, mul_)
    AKR_ARRAY_OP(/, div_)
    AKR_ARRAY_OP(&, and_)
    AKR_ARRAY_OP(|, or_)
    AKR_ARRAY_OP(&&, and_)
    AKR_ARRAY_OP(||, or_)
    AKR_ARRAY_OP(^, xor_)
#undef AKR_ARRAY_OP
} // namespace akari

using namespace akari;

int main() {
    Array<int, 3> a(1, 2, 4);
    Array<int, 3> c(77);
    auto b = c + a;
    printf("%s\n", typeid(ArrayBaseImplBase<int, 4, minmaxpow2(4)>).name());
    // static_assert(adjusted_array_size(3) == 4);
    // static_assert(sizeof(a) == sizeof(float) * 4);
    (void)a;
    printf("%zd %d %d %d\n", sizeof(a), c[0], c[1], c[2]);
    printf("%zd %d %d %d\n", sizeof(a), a[0], a[1], a[2]);
    printf("%zd %d %d %d\n", sizeof(a), b[0], b[1], b[2]);
}