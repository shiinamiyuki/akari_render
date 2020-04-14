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
#include <initializer_list>
#include <type_traits>
#include <xmmintrin.h>

namespace Akari {
#define AKR_INLINE    __forceline
#define AKR_ISA_NONE  0
#define AKR_ISA_SSE   1
#define AKR_ISA_SSE42 2
#define AKR_ISA_AVX   3
#define AKR_ISA_AVX2  4

    __forceinline __m128i _mm_cmple_epi32(__m128i a, __m128i b) {
        return _mm_or_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, b));
    }
    __forceinline __m128i _mm_cmpge_epi32(__m128i a, __m128i b) {
        return _mm_or_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, b));
    }
    __forceinline __m128i _mm_cmpneq_epi32(__m128i a, __m128i b) {
        return _mm_xor_epi32(_mm_cmpeq_epi32(a, b), _mm_set1_epi8(0xff));
    }
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
#define AKR_SIMD_STRUCT_(Ty, N, Reg, Self, Half)                                                                       \
    union {                                                                                                            \
        Reg m;                                                                                                         \
        Ty s[N];                                                                                                       \
        struct {                                                                                                       \
            Half lo;                                                                                                   \
            Half hi;                                                                                                   \
        };                                                                                                             \
    };                                                                                                                 \
    Self() = default;                                                                                                  \
    Self(Reg m) : m(m) {}                                                                                              \
    Self &operator=(Reg v) {                                                                                           \
        m = v;                                                                                                         \
        return *this;                                                                                                  \
    }                                                                                                                  \
    operator Reg() const { return m; }

    struct simd_f32x4 {
        AKR_SIMD_STRUCT_(float, 4, __m128, simd_f32x4, __m64)
        simd_f32x4(float v0, float v1, float v2, float v3) : m(_mm_set_ps(v3, v2, v1, v0)) {}
        simd_f32x4(float v) : m(_mm_set1_ps(v)) {}
    };
    struct simd_f32x8 {
        AKR_SIMD_STRUCT_(float, 8, __m256, simd_f32x8, __m128)
        simd_f32x8(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7)
            : m(_mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0)) {}
        simd_f32x8(float v) : m(_mm256_set1_ps(v)) {}
    };
    struct simd_i32x4 {
        AKR_SIMD_STRUCT_(int32_t, 4, __m128i, simd_i32x4, __m64)
        simd_i32x4(int32_t v0, int32_t v1, int32_t v2, int32_t v3) : m(_mm_set_epi32(v3, v2, v1, v0)) {}
        simd_i32x4(int32_t v) : m(_mm_set1_epi32(v)) {}
    };
    struct simd_i32x8 {
        AKR_SIMD_STRUCT_(int32_t, 8, __m256i, simd_i32x8, __m128i)
        simd_i32x8(int32_t v0, int32_t v1, int32_t v2, int32_t v3, int32_t v4, int32_t v5, int32_t v6, int32_t v7)
            : m(_mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0)) {}
        simd_i32x8(int32_t v) : m(_mm256_set1_epi32(v)) {}
        simd_i32x8(__m128i lo, __m128i hi) : m(_mm256_set_m128i(hi, lo)) {}
    };
    template <typename T, size_t N> struct array_simd_reg_;
    template <> struct array_simd_reg_<float, 4> { using type = simd_f32x4; };
    template <> struct array_simd_reg_<int32_t, 4> { using type = simd_i32x4; };
    template <> struct array_simd_reg_<float, 8> { using type = simd_f32x8; };
    template <> struct array_simd_reg_<int32_t, 8> { using type = simd_i32x8; };
    template <typename T, size_t N> using array_simd_reg = typename array_simd_reg_<T, N>::type;
    template <typename T> struct is_simd_type : std::false_type {};
    template <> struct is_simd_type<float> : std::true_type {};
    template <> struct is_simd_type<int32_t> : std::true_type {};
#define AKR_STORAGE_ASSERT_ALIGNED() static_assert(!is_simd_type<T>::value || N % 4 == 0)
#define AKR_REINTERPRET_ARRAY(Ty)                                                                                      \
    const Ty *raw() const { return reinterpret_cast<const Ty *>(this); }                                               \
    Ty *raw() { return reinterpret_cast<Ty *>(this); }                                                                 \
    const Ty &operator[](size_t i) const { return raw()[i]; }                                                          \
    Ty &operator[](size_t i) { return raw()[i]; }

#define AKR_ARRAY_GENERATE_OPERATOR_(assign_op, func)                                                                  \
    Self &operator assign_op(const Self &rhs) {                                                                        \
        auto tmp = this->func(rhs);                                                                                    \
        *this = tmp;                                                                                                   \
        return *this;                                                                                                  \
    }

    template <typename T, size_t N> struct array_mask;
    template <typename T, size_t N> struct get_array_mask { using type = array_mask<int32_t, N>; };
    template <size_t N> struct get_array_mask<float, N> { using type = array_mask<float, N>; };
    template <size_t N> struct get_array_mask<int32_t, N> { using type = array_mask<int32_t, N>; };
#define AKR_ARRAY_MASK_GENERATE_OPERATORS()                                                                            \
    AKR_ARRAY_GENERATE_OPERATOR_(&=, _and)                                                                             \
    AKR_ARRAY_GENERATE_OPERATOR_(|=, _or)                                                                              \
    AKR_ARRAY_GENERATE_OPERATOR_(^=, _xor)

    template <typename T, size_t N> struct array_mask {
        AKR_STORAGE_ASSERT_ALIGNED();
        AKR_REINTERPRET_ARRAY(T)
        using type1 = array_mask<T, 8>;
        using type2 = array_mask<T, N - 8>;
        using Self = array_mask;
        AKR_ARRAY_MASK_GENERATE_OPERATORS()
        type1 mask1;
        type2 mask2;
        array_mask() = default;
        array_mask(const type1 &array1, const type2 &array2) : mask1(array1), mask2(array2) {}
#define AKR_ARRAY_MASK_RECURSIVE_FUNC_1(Ret, func)                                                                     \
    Ret func(const Self &other) const { return Ret(mask1.func(other.mask1), mask2.func(other.mask2)); }
#define AKR_ARRAY_MASK_RECURSIVE_FUNC_0(Ret, func)                                                                     \
    Ret func() const { return Ret(mask1.func(), mask2.func()); }
        AKR_ARRAY_MASK_RECURSIVE_FUNC_1(Self, _and)
        AKR_ARRAY_MASK_RECURSIVE_FUNC_1(Self, _or)
        AKR_ARRAY_MASK_RECURSIVE_FUNC_1(Self, _xor)
        AKR_ARRAY_MASK_RECURSIVE_FUNC_0(Self, _not)
    };
    template <typename T> struct array_mask<T, 8> : simd_f32x8 {
        using simd_f32x8::simd_f32x8;
        using Self = array_mask;
        AKR_ARRAY_MASK_GENERATE_OPERATORS()
        Self _and(const Self &_other) const { return _mm256_and_ps(m, _other.m); }
        Self _or(const Self &_other) const { return _mm256_or_ps(m, _other.m); }
        Self _xor(const Self &_other) const { return _mm256_xor_ps(m, _other.m); }
        Self _not(const Self &_other) const { return _mm256_xor_ps(m, (__m256)_mm256_set1_epi8(0xff)); }
    };
    template <> struct array_mask<int, 4> : simd_i32x4 {
        using simd_i32x4::simd_i32x4;
        using Self = array_mask;
        AKR_REINTERPRET_ARRAY(int)
        AKR_ARRAY_MASK_GENERATE_OPERATORS()
        Self _and(const Self &_other) const { return (__m128i)_mm_and_ps((__m128)m, (__m128)_other.m); }
        Self _or(const Self &_other) const { return (__m128i)_mm_or_ps((__m128)m, (__m128)_other.m); }
        Self _xor(const Self &_other) const { return (__m128i)_mm_xor_ps((__m128)m, (__m128)_other.m); }
        Self _not() const { return _mm_xor_epi32(m, _mm_set1_epi8(0xff)); }
    };
    template <> struct array_mask<int, 8> {
        using type1 = array_mask<int, 4>;
        using type2 = type1;
        type1 mask1;
        type2 mask2;
        using Self = array_mask;
        AKR_REINTERPRET_ARRAY(int)
        AKR_ARRAY_MASK_GENERATE_OPERATORS()
        array_mask() = default;
        array_mask(__m256i m) {
            simd_i32x8 i32X8(m);
            mask1 = type1(i32X8.lo);
            mask2 = type1(i32X8.hi);
        }
        array_mask(const type1 &array1, const type2 &array2) : mask1(array1), mask2(array2) {}
        AKR_ARRAY_MASK_RECURSIVE_FUNC_1(Self, _and)
        AKR_ARRAY_MASK_RECURSIVE_FUNC_1(Self, _or)
        AKR_ARRAY_MASK_RECURSIVE_FUNC_1(Self, _xor)
        AKR_ARRAY_MASK_RECURSIVE_FUNC_0(Self, _not)
    };
    template <> struct array_mask<float, 4> : simd_f32x4 {
        using simd_f32x4::simd_f32x4;
        using Self = array_mask;
        AKR_REINTERPRET_ARRAY(float)
        AKR_ARRAY_MASK_GENERATE_OPERATORS()
        Self _and(const Self &_other) const { return _mm_and_ps(m, _other.m); }
        Self _or(const Self &_other) const { return _mm_or_ps(m, _other.m); }
        Self _xor(const Self &_other) const { return _mm_xor_ps(m, _other.m); }
        Self _not() const { return _mm_xor_ps(m, (__m128)_mm_set1_epi8(0xff)); }
    };
    template <> struct array_mask<float, 8> : simd_f32x8 {
        using simd_f32x8::simd_f32x8;
        using Self = array_mask;
        AKR_REINTERPRET_ARRAY(float)
        AKR_ARRAY_MASK_GENERATE_OPERATORS()
        Self _and(const Self &_other) const { return _mm256_and_ps(m, _other.m); }
        Self _or(const Self &_other) const { return _mm256_or_ps(m, _other.m); }
        Self _xor(const Self &_other) const { return _mm256_xor_ps(m, _other.m); }
        Self _not() const { return _mm256_xor_ps(m, (__m256)_mm256_set1_epi8(0xff)); }
    };
    template <typename T> struct SIMDLane {
        constexpr static size_t value = (std::is_same_v<T, int32_t> || std::is_same_v<T, float>) ? 8 : 1;
    };
    template <typename T, size_t N, typename Derived = void> struct array_storage_base;
    template <typename T, size_t N> struct _to_array_storage {
        using type = std::conditional_t<N == 1, T, array_storage_base<T, N>>;
    };
    template <typename T, size_t N> using _to_array_storage_t = typename _to_array_storage<T, N>::type;
    template <typename T, size_t N, typename Derived> struct array_storage_base {
        AKR_STORAGE_ASSERT_ALIGNED();
        constexpr static size_t lane = SIMDLane<T>::value;
        using type1 = _to_array_storage_t<T, lane>;
        using type2 = _to_array_storage_t<T, N - lane>;
        using Self = array_storage_base;
        using Mask = array_mask<int32_t, N>;
        using ReturnT = std::conditional_t<std::is_same_v<Derived, void>, Self, Derived>;
        type1 array1;
        type2 array2;
        array_storage_base() = default;
        array_storage_base(const T &v) : array1(v), array2(v) {}
        array_storage_base(const type1 &array1, const type2 &array2) : array1(array1), array2(array2) {}

//        Derived &derived() { return static_cast<Derived &>(*this); }
#define AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(Ret, func)                                                                  \
    Ret func(const Self &other) const { return Ret(array1.func(other.array1), array2.func(other.array2)); }
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(ReturnT, _add)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(ReturnT, _sub)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(ReturnT, _mul)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(ReturnT, _div)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(Mask, _lt)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(Mask, _gt)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(Mask, _le)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(Mask, _ge)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(Mask, _eq)
        AKR_ARRAY_STORAGE_RECURSIVE_FUNC_1(Mask, _ne)
    };

    template <typename Derived> struct array_storage_base<float, 8, Derived> : simd_f32x8 {
        using simd_f32x8::simd_f32x8;
        using Self = array_storage_base;
        using Mask = array_mask<int32_t, 8>;
        using ReturnT = std::conditional_t<std::is_same_v<Derived, void>, Self, Derived>;
        ReturnT _add(const Self &_other) const { return ReturnT(_mm256_add_ps(this->m, _other.m)); }
        ReturnT _sub(const Self &_other) const { return ReturnT(_mm256_sub_ps(this->m, _other.m)); }
        ReturnT _mul(const Self &_other) const { return ReturnT(_mm256_mul_ps(this->m, _other.m)); }
        ReturnT _div(const Self &_other) const { return ReturnT(_mm256_div_ps(this->m, _other.m)); }
        Mask _lt(const Self &_other) const { return Mask((__m256i)_mm256_cmp_ps(this->m, _other.m, _CMP_LT_OQ)); }
        Mask _gt(const Self &_other) const { return Mask((__m256i)_mm256_cmp_ps(this->m, _other.m, _CMP_GT_OQ)); }
        Mask _le(const Self &_other) const { return Mask((__m256i)_mm256_cmp_ps(this->m, _other.m, _CMP_LE_OQ)); }
        Mask _ge(const Self &_other) const { return Mask((__m256i)_mm256_cmp_ps(this->m, _other.m, _CMP_GE_OQ)); }
        Mask _eq(const Self &_other) const { return Mask((__m256i)_mm256_cmp_ps(this->m, _other.m, _CMP_EQ_OQ)); }
        Mask _ne(const Self &_other) const { return Mask((__m256i)_mm256_cmp_ps(this->m, _other.m, _CMP_NEQ_OQ)); }
    };
    template <typename Derived> struct array_storage_base<float, 4, Derived> : simd_f32x4 {
        using Self = array_storage_base;
        using Mask = array_mask<int32_t, 4>;
        using simd_f32x4::simd_f32x4;
        using ReturnT = std::conditional_t<std::is_same_v<Derived, void>, Self, Derived>;
        ReturnT _add(const Self &_other) const { return Self(_mm_add_ps(this->m, _other.m)); }
        ReturnT _sub(const Self &_other) const { return Self(_mm_sub_ps(this->m, _other.m)); }
        ReturnT _mul(const Self &_other) const { return Self(_mm_mul_ps(this->m, _other.m)); }
        ReturnT _div(const Self &_other) const { return Self(_mm_div_ps(this->m, _other.m)); }
        Mask _lt(const Self &_other) const { return Mask((__m128i)_mm_cmp_ps(this->m, _other.m, _CMP_LT_OQ)); }
        Mask _gt(const Self &_other) const { return Mask((__m128i)_mm_cmp_ps(this->m, _other.m, _CMP_GT_OQ)); }
        Mask _le(const Self &_other) const { return Mask((__m128i)_mm_cmp_ps(this->m, _other.m, _CMP_LE_OQ)); }
        Mask _ge(const Self &_other) const { return Mask((__m128i)_mm_cmp_ps(this->m, _other.m, _CMP_GE_OQ)); }
        Mask _eq(const Self &_other) const { return Mask((__m128i)_mm_cmp_ps(this->m, _other.m, _CMP_EQ_OQ)); }
        Mask _ne(const Self &_other) const { return Mask((__m128i)_mm_cmp_ps(this->m, _other.m, _CMP_NEQ_OQ)); }
    };

    template <typename Derived> struct array_storage_base<int, 8, Derived> : simd_i32x8 {
        using simd_i32x8::simd_i32x8;
        using Self = array_storage_base;
        using Mask = array_mask<int32_t, 8>;
        Self _add(const Self &_other) const { return Self(_mm256_add_epi32(this->m, _other.m)); }
        Self _sub(const Self &_other) const { return Self(_mm256_sub_epi32(this->m, _other.m)); }
        Self _mul(const Self &_other) const { return Self(_mm256_mul_epi32(this->m, _other.m)); }
        Self _div(const Self &_other) const {
            return Self(this->s[0] / _other.s[0], this->s[1] / _other.s[1], this->s[2] / _other.s[2],
                        this->s[3] / _other.s[3], this->s[4] / _other.s[4], this->s[5] / _other.s[5],
                        this->s[6] / _other.s[6], this->s[7] / _other.s[7]);
        }
        Mask _lt(const Self &_other) const {
            return Mask(_mm_cmplt_epi32(this->lo, _other.lo), _mm_cmplt_epi32(this->hi, _other.hi));
        }
        Mask _gt(const Self &_other) const {
            return Mask(_mm_cmpgt_epi32(this->lo, _other.lo), _mm_cmpgt_epi32(this->hi, _other.hi));
        }
        Mask _le(const Self &_other) const {
            return Mask(_mm_cmple_epi32(this->lo, _other.lo), _mm_cmple_epi32(this->hi, _other.hi));
        }
        Mask _ge(const Self &_other) const {
            return Mask(_mm_cmpge_epi32(this->lo, _other.lo), _mm_cmpge_epi32(this->hi, _other.hi));
        }
        Mask _eq(const Self &_other) const {
            return Mask(_mm_cmpeq_epi32(this->lo, _other.lo), _mm_cmpeq_epi32(this->hi, _other.hi));
        }
        Mask _ne(const Self &_other) const {
            return Mask(_mm_cmpneq_epi32(this->lo, _other.lo), _mm_cmpneq_epi32(this->hi, _other.hi));
        }
    };
    template <typename Derived> struct array_storage_base<int, 4, Derived> : simd_i32x4 {
        using Self = array_storage_base;
        using Mask = array_mask<int32_t, 4>;
        using simd_i32x4::simd_i32x4;
        Self _add(const Self &_other) const { return Self(_mm_add_epi32(this->m, _other.m)); }
        Self _sub(const Self &_other) const { return Self(_mm_sub_epi32(this->m, _other.m)); }
        Self _mul(const Self &_other) const { return Self(_mm_mul_epi32(this->m, _other.m)); }
        Self _div(const Self &_other) const {
            return Self(this->s[0] / _other.s[0], this->s[1] / _other.s[1], this->s[2] / _other.s[2],
                        this->s[3] / _other.s[3]);
        }
        Mask _lt(const Self &_other) const { return Mask(_mm_cmplt_epi32(this->m, _other.m)); }
        Mask _gt(const Self &_other) const { return Mask(_mm_cmpgt_epi32(this->m, _other.m)); }
        Mask _le(const Self &_other) const { return Mask(_mm_cmple_epi32(this->m, _other.m)); }
        Mask _ge(const Self &_other) const { return Mask(_mm_cmpge_epi32(this->m, _other.m)); }
        Mask _eq(const Self &_other) const { return Mask(_mm_cmpeq_epi32(this->m, _other.m)); }
        Mask _ne(const Self &_other) const { return Mask(_mm_cmpneq_epi32(this->m, _other.m)); }
    };

    template <size_t N> struct bit_mask {
        static_assert(N < 32);
        bit_mask(uint32_t m) : mask(m) {}
#define AKR_GEN_BIT_MASK_OPERATOR(op, assign_op)                                                                       \
    bit_mask &operator assign_op(const bit_mask &rhs) {                                                                \
        mask assign_op rhs.mask;                                                                                       \
        return *this;                                                                                                  \
    }                                                                                                                  \
    bit_mask operator op(const bit_mask &rhs) const { return mask op rhs.mask; }
        AKR_GEN_BIT_MASK_OPERATOR(&, &=)
        AKR_GEN_BIT_MASK_OPERATOR(|, |=)
        AKR_GEN_BIT_MASK_OPERATOR(^, ^=)
        operator bool() const { return mask; }
        uint32_t mask;
    };

    template <typename T, size_t N, typename SFINAE = void> struct aligned_array_length {
        constexpr static size_t value = N;
    };
    template <typename T, size_t N> struct aligned_array_length<T, N, std::enable_if_t<is_simd_type<T>::type>> {
        constexpr static size_t value = align_to<N, 4>;
    };
    template <typename T, size_t N> struct array : array_storage_base<T, N, array<T, N>> {
        using Base = array_storage_base<T, N, array<T, N>>;
        using array_storage_base<T, N, array<T, N>>::array_storage_base;
        using Self = array;
        AKR_REINTERPRET_ARRAY(T)
#define AKR_ARRAY_GENERATE_OPERATORS()                                                                                 \
    AKR_ARRAY_GENERATE_OPERATOR_(+=, _add)                                                                             \
    AKR_ARRAY_GENERATE_OPERATOR_(-=, _sub)                                                                             \
    AKR_ARRAY_GENERATE_OPERATOR_(*=, _mul)                                                                             \
    AKR_ARRAY_GENERATE_OPERATOR_(/=, _div)
        AKR_ARRAY_GENERATE_OPERATORS()
    };
    template <typename T1, typename T2> struct expr_t_ { using type = T1; };
    template <> struct expr_t_<int, float> { using type = float; };
    template <> struct expr_t_<float, float> { using type = float; };
    template <> struct expr_t_<float, int> { using type = float; };
    template <typename T1, typename T2> using expr_t = typename expr_t_<T1, T2>::type;
#define AKR_ARRAY_MASK_GENERATOR_BINOP(op, func)                                                                       \
    template <typename T, typename U, size_t N>                                                                        \
    inline decltype(auto) operator op(const array_mask<T, N> &a, const array_mask<U, N> &b) {                          \
        using ExprT = expr_t<T, U>;                                                                                    \
        if constexpr (std::is_same_v<ExprT, T>) {                                                                      \
            auto tmpB = array_mask<T, N>(a);                                                                           \
            return a.func(b);                                                                                          \
        } else {                                                                                                       \
            auto tmpA = array_mask<U, N>(a);                                                                           \
            return a.func(b);                                                                                          \
        }                                                                                                              \
    }
#define AKR_ARRAY_GENERATOR_BINOP(op, func)                                                                            \
    template <typename T, typename U, size_t N>                                                                        \
    inline decltype(auto) operator op(const array<T, N> &a, const array<U, N> &b) {                                    \
        using ExprT = expr_t<T, U>;                                                                                    \
        if constexpr (std::is_same_v<ExprT, T>) {                                                                      \
            auto tmpB = array<T, N>(a);                                                                                \
            return a.func(b);                                                                                          \
        } else {                                                                                                       \
            auto tmpA = array<U, N>(a);                                                                                \
            return a.func(b);                                                                                          \
        }                                                                                                              \
    }
#define AKR_ARRAY_GEN_BINOPS()                                                                                         \
    AKR_ARRAY_GENERATOR_BINOP(+, _add)                                                                                 \
    AKR_ARRAY_GENERATOR_BINOP(-, _sub)                                                                                 \
    AKR_ARRAY_GENERATOR_BINOP(*, _mul)                                                                                 \
    AKR_ARRAY_GENERATOR_BINOP(/, _div)                                                                                 \
    AKR_ARRAY_GENERATOR_BINOP(<, _lt)                                                                                  \
    AKR_ARRAY_GENERATOR_BINOP(>, _gt)                                                                                  \
    AKR_ARRAY_GENERATOR_BINOP(<=, _le)                                                                                 \
    AKR_ARRAY_GENERATOR_BINOP(>=, _ge)                                                                                 \
    AKR_ARRAY_GENERATOR_BINOP(==, _eq)                                                                                 \
    AKR_ARRAY_GENERATOR_BINOP(!=, _ne)
    AKR_ARRAY_GEN_BINOPS()

    AKR_ARRAY_MASK_GENERATOR_BINOP(&, _and)
    AKR_ARRAY_MASK_GENERATOR_BINOP(|, _or)
    AKR_ARRAY_MASK_GENERATOR_BINOP(^, _xor)

    template <typename T, size_t N> array_mask<T, N> operator~(const array_mask<T, N> &mask) { return mask._not(); }

} // namespace Akari
int main() {
    using namespace Akari;
    array<float, 32> v1(1);
    array<float, 32> v2(2);
    for (int i = 0; i < 32; i++) {
        v1[i] = 2 * i + 1;
        v2[i] = 3 * i + 2;
    }
    auto mask = (v1 < 50)
    mask &= mask;
    for (int i = 0; i < 32; i++) {
        printf("%f %f %d\n", v3[i], v1[i], mask[i]);
    }
    //    simd_array<float, 32> a, b;
    //    for (int i = 0; i < 32; i++) {
    //        a[i] = 2 * i + 1;
    //        b[i] = 3 * i + 2;
    //    }
    //    a = a + b;
    //    auto mask = array_operator_lt<float, 32>::apply(a, b);
    //    for (int i = 0; i < 32; i++) {
    //        printf("%f %f %d\n", a[i], b[i], mask[i]);
    //    }
    //    auto c = select(~(a<100.0f & a> 50.0f), a, b);
    //    for (int i = 0; i < 32; i++) {
    //        printf("%f %f %f %d\n", a[i], b[i], c[i], mask[i]);
    //    }
    //    using namespace glm;
    //    vec<4, simd_array<float, 32>, defaultp> v1(1), v2(3);
    //    auto v3 = v1 + v2;
}
