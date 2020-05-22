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

#include <immintrin.h>
#include <type_traits>
#include <array>
namespace akari {
    template <typename T, size_t N> struct ArrayStorage {
        using element_type = T;
        using Self = ArrayStorage;
        static constexpr bool is_specialized = false;
        alignas(alignof(T)) std::array<T, N> s;
        template <typename F> __forceinline ArrayStorage apply(F &&f) const {
            ArrayStorage tmp;
            for (size_t i = 0; i < N; i++) {
                tmp.s[i] = f(s[i]);
            }
            return tmp;
        }
#define AKR_ARRAY_DEFAULT_OP(op, func)                                                                                 \
    __forceinline Self func(const Self &rhs) const {                                                                   \
        ArrayStorage tmp;                                                                                              \
        for (size_t i = 0; i < N; i++) {                                                                               \
            tmp.s[i] = f(s[i] op rhs.s[i]);                                                                            \
        }                                                                                                              \
        return tmp;                                                                                                    \
    }
        AKR_ARRAY_DEFAULT_OP(+, add_)
        AKR_ARRAY_DEFAULT_OP(-, sub_)
        AKR_ARRAY_DEFAULT_OP(*, mul_)
        AKR_ARRAY_DEFAULT_OP(/, div_)
#undef AKR_ARRAY_DEFAULT_OP
    };
   
    template <> struct ArrayStorage<float, 4> {
        using element_type = float;
        static constexpr bool is_specialized = true;
        using Self = ArrayStorage;
        union {
            __m128 f32x4;
            __m128i i32x4;
            float s[4];
        };
        ArrayStorage()=default;
        ArrayStorage(float x):f32x4(_mm_set1_ps(x)){}
        ArrayStorage(__m128 x) : f32x4(x) {}
#define AKR_ARRAY_OP(func, intrin)                                                                                     \
    __forceinline Self func(const Self &rhs) const { return Self(intrin(this->f32x4, rhs.f32x4)); }
        AKR_ARRAY_OP(add_, _mm_add_ps)
        AKR_ARRAY_OP(sub_, _mm_sub_ps)
        AKR_ARRAY_OP(mul_, _mm_mul_ps)
        AKR_ARRAY_OP(div_, _mm_div_ps)
        AKR_ARRAY_OP(min_, _mm_min_ps)
        AKR_ARRAY_OP(max_, _mm_max_ps)
#undef AKR_ARRAY_OP
    };

    template <> struct ArrayStorage<float, 8> {
        using element_type = float;
        using Self = ArrayStorage;
        static constexpr bool is_specialized = true;
        union {
            __m256 f32x8;
            __m256i i32x8;
            float s[8];
        };
        ArrayStorage() = default;
        ArrayStorage(float x):f32x8(_mm256_set1_ps(x)){}
        ArrayStorage(__m256 x) : f32x8(x) {}
#define AKR_ARRAY_OP(func, intrin)                                                                                     \
    __forceinline Self func(const Self &rhs) const { return Self(intrin(this->f32x8, rhs.f32x8)); }
        AKR_ARRAY_OP(add_, _mm256_add_ps)
        AKR_ARRAY_OP(sub_, _mm256_sub_ps)
        AKR_ARRAY_OP(mul_, _mm256_mul_ps)
        AKR_ARRAY_OP(div_, _mm256_div_ps)
        AKR_ARRAY_OP(min_, _mm256_min_ps)
        AKR_ARRAY_OP(max_, _mm256_max_ps)
#undef AKR_ARRAY_OP
    };
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
         if(x <= 2){
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
        ArrayStoragePair(const T & a):lo(a),hi(a){}
        ArrayStoragePair(const Array1 &a, const Array2 &b) : lo(a), hi(b) {}
#define AKR_ARRAY_OP(func)                                                                                             \
    __forceinline Self func(const Self &rhs) const { return Self(lo.func(rhs.lo), hi.func(rhs.hi)); }
        AKR_ARRAY_OP(add_)
        AKR_ARRAY_OP(sub_)
        AKR_ARRAY_OP(mul_)
        AKR_ARRAY_OP(div_)
        AKR_ARRAY_OP(min_)
        AKR_ARRAY_OP(max_)
#undef AKR_ARRAY_OP
    };
    template <typename T, size_t N, size_t M>struct ArrayStoragePair_Indirect{
        using type = ArrayStoragePair<ArrayStorage<T, N>, ArrayStorage<T, N - M>>;
    };
     template <typename T, size_t N>struct ArrayStorage_Indirect{
        using type = ArrayStorage<T, N>;
    };
    template <typename T, size_t N, size_t M>
    using ArrayBaseImplBase = typename std::conditional_t<
        ArrayStorage<T, N>::is_specialized,
        ArrayStorage_Indirect<T, N>,
        std::conditional_t< N == M,  ArrayStoragePair_Indirect<T, N / 2, N / 2>,
        ArrayStoragePair_Indirect<T, M, N - M>>>::type;
    template <typename T, size_t N, size_t M> struct ArrayBaseImpl : ArrayBaseImplBase<T, N, M> {
        using Self = ArrayBaseImpl;
        using Base = ArrayBaseImplBase<T, N, M>;
        ArrayBaseImpl(const T & v):Base(v){}
#define AKR_ARRAY_OP(func)                                                                                             \
    __forceinline Self func(const Self &rhs) const { return Self(static_cast<const Base *>(this)->func(rhs)); }
        AKR_ARRAY_OP(add_)
        AKR_ARRAY_OP(sub_)
        AKR_ARRAY_OP(mul_)
        AKR_ARRAY_OP(div_)
        AKR_ARRAY_OP(min_)
        AKR_ARRAY_OP(max_)
#undef AKR_ARRAY_OP
    };
    template <typename T, size_t N, typename Derived> struct ArrayBaseCommon : ArrayBaseImpl<T, N> {
        using Self = ArrayBaseCommon;
        using Base = ArrayBaseImpl<T, N>;
        using Base::ArrayBaseImpl;
#define AKR_ARRAY_OP(func)                                                                                             \
    __forceinline Derived func(const Self &rhs) const { return Self(static_cast<const Base *>(this)->func(rhs)); }
        AKR_ARRAY_OP(add_)
        AKR_ARRAY_OP(sub_)
        AKR_ARRAY_OP(mul_)
        AKR_ARRAY_OP(div_)
        AKR_ARRAY_OP(min_)
        AKR_ARRAY_OP(max_)
#undef AKR_ARRAY_OP
#define AKR_ARRAY_IOP(op, func)                                                                                        \
    __forceinline Derived &operator op(const Self &rhs) {                                                              \
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
        T &operator[](size_t idx) { return this->s[idx]; }
        const T &operator[](size_t idx) const { return this->s[idx]; }
        template <size_t Idx> T &at() {
            static_assert(Idx < N);
            return this->s[Idx];
        }
        template <size_t Idx> const T &at() const {
            static_assert(Idx < N);
            return this->s[Idx];
        }
    };
    template <typename T, size_t N, size_t _M> struct Array : ArrayBase<T, _M, Array<T, N, _M>> {
        using Base = ArrayBase<T, _M, Array<T, N, _M>>;
        using Base::ArrayBase;
    };
} // namespace akari

using namespace akari;

int main() {
    Array<float, 3> a(1.0f);
    // static_assert(adjusted_array_size(3) == 4);
    // static_assert(sizeof(a) == sizeof(float) * 4);
    (void)a;
    printf("%zd\n", sizeof(a));
}