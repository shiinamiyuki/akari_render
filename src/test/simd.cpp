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

#include <cstdio>
#include <cstring>
#include <array>
#include <immintrin.h>
#include <type_traits>
#include <cstdint>
#include <cmath>
namespace akari {
    using std::min;
    using std::max;
    using std::floor;
    using std::ceil;
    template <typename T, int N, bool isMask = false> struct Array;
    template <typename T, int N> constexpr bool vector_recursive_is_base_case() { return N <= 4; }
    template <typename T, int N, bool isMask, typename Derived, typename = void> struct ArrayImpl {};

    template <typename V> struct ArrayMask;
    template <typename T, int N, bool isMask> struct ArrayMask<Array<T, N, isMask>> { using type = Array<T, N, true>; };
    template <typename T, int N, typename Derived>
    struct ArrayImpl<T, N, false, Derived, std::enable_if_t<vector_recursive_is_base_case<T, N>()>> {
        std::array<T, N> arr;
        ArrayImpl() = default;
        ArrayImpl(T v) {
            for (int i = 0; i < N; i++) {
                arr[i] = v;
            }
        }
        ArrayImpl(const std::array<T, N> &arr) : arr(arr) {}
#define AKR_FUNC1(func, op)                                                                                            \
    Derived func(const Derived &rhs) const {                                                                           \
        std::array<T, N> r;                                                                                            \
        for (int i = 0; i < N; i++) {                                                                                  \
            r[i] = arr[i] op rhs.arr[i];                                                                                 \
        }                                                                                                              \
        return Derived(r);                                                                                             \
    }
        AKR_FUNC1(add, +)
        AKR_FUNC1(sub, -)
        AKR_FUNC1(mul, *)
        AKR_FUNC1(div, /)
        AKR_FUNC1(mod, %)
        AKR_FUNC1(and_, &)
        AKR_FUNC1(or_, |)
        AKR_FUNC1(xor_, ^)
#undef AKR_FUNC1
#define AKR_FUNC0(func, f)                                                                                                \
    Derived func() const {                                                                                             \
        std::array<T, N> r;                                                                                            \
        for (int i = 0; i < N; i++) {                                                                                  \
            r[i] = f(arr[i]);                                                                                     \
        }                                                                                                              \
        return Derived(r);                                                                                             \
    }
        AKR_FUNC0(floor_,floor)
        AKR_FUNC0(ceil_, ceil)
#define AKR_FUNC1(func, f)                                                                                                \
    Derived func(const Derived &rhs) const {                                                                           \
        std::array<T, N> r;                                                                                            \
        for (int i = 0; i < N; i++) {                                                                                  \
            r[i] = f(arr[i], rhs.arr[i]);                                                                         \
        }                                                                                                              \
        return Derived(r);                                                                                             \
    }
        AKR_FUNC1(min_,min)
        AKR_FUNC1(max_,max)
#undef AKR_FUNC1

        using Mask = typename ArrayMask<Derived>::type;
#define AKR_FUNC1(func, op)                                                                                            \
    Mask func(const Derived &rhs) const {                                                                              \
        Mask m;                                                                                                        \
        for (int i = 0; i < N; i++) {                                                                                  \
            m.arr[i] = arr[i] op rhs.arr[i];                                                                           \
        }                                                                                                              \
        return m;                                                                                                      \
    }
        AKR_FUNC1(lt, <)
        AKR_FUNC1(gt, >)
        AKR_FUNC1(le, <=)
        AKR_FUNC1(ge, >=)
        AKR_FUNC1(ne, !=)
        AKR_FUNC1(eq, ==)
#undef AKR_FUNC1
        static Derived select(const Mask &m, const Derived &x, const Derived &y) {
            std::array<T, N> r;
            for (int i = 0; i < N; i++) {
                r[i] = m.arr[i] ? x.arr[i] : y.arr[i];
            }
            return Derived(r);
        }
    };

    template <typename T, int N, typename Derived>
    struct ArrayImpl<T, N, true, Derived, std::enable_if_t<vector_recursive_is_base_case<T, N>()>> {
        std::array<uint32_t, N> arr;
        ArrayImpl() = default;
        ArrayImpl(const std::array<uint32_t, N> &arr) : arr(arr) {}
#define AKR_FUNC1(func, op)                                                                                            \
    Derived func(const Derived &rhs) const {                                                                           \
        std::array<T, N> r;                                                                                            \
        for (int i = 0; i < N; i++) {                                                                                  \
            r[i] = arr[i] op rhs.arr[i];                                                                                 \
        }                                                                                                              \
        return Derived(r);                                                                                             \
    }
        AKR_FUNC1(and_, &)
        AKR_FUNC1(or_, |)
        AKR_FUNC1(xor_, ^)

        static Derived select(const Derived &m, const Derived &x, const Derived &y) {
            std::array<uint32_t, N> r;
            for (int i = 0; i < N; i++) {
                r[i] = m[i] ? x.arr[i] : y.arr[i];
            }
            return Derived(r);
        }
    };
    template <typename Derived> struct maskf32x4 {
        __m128 m;
        const maskf32x4 &_this() const { return static_cast<const maskf32x4 &>(*this); }
        maskf32x4(float v) : m(_mm_set_ps1(v)) {}
        maskf32x4() = default;
        maskf32x4(__m128 m) : m(m) {}
#define AKR_F32X4_FUNC(func, intrinsic)                                                                                \
    Derived func(const Derived &rhs) const { return Derived(intrinsic(m, rhs._this().m)); }
        AKR_F32X4_FUNC(and_, _mm_and_ps)
        AKR_F32X4_FUNC(or_, _mm_or_ps)
        AKR_F32X4_FUNC(xor_, _mm_xor_ps)
#undef AKR_F32X4_FUNC
        static Derived select(const Derived &m, const Derived &x, const Derived &y) {
            return Derived(_mm_blendv_ps(y._this().m, x._this().m, m.m));
        }
    };

    template <typename Derived> struct f32x4 {
        __m128 m;
        const f32x4 &_this() const { return static_cast<const f32x4 &>(*this); }
        f32x4(float v) : m(_mm_set_ps1(v)) {}
        f32x4() = default;
        f32x4(__m128 m) : m(m) {}
#define AKR_F32X4_FUNC(func, intrinsic)                                                                                \
    Derived func(const Derived &rhs) const { return Derived(intrinsic(m, rhs._this().m)); }
        AKR_F32X4_FUNC(add, _mm_add_ps)
        AKR_F32X4_FUNC(sub, _mm_sub_ps)
        AKR_F32X4_FUNC(mul, _mm_mul_ps)
        AKR_F32X4_FUNC(div, _mm_div_ps)
        AKR_F32X4_FUNC(min_, _mm_min_ps)
        AKR_F32X4_FUNC(max_, _mm_max_ps)
#undef AKR_F32X4_FUNC
#define AKR_F32X4_FUNC(func, intrinsic)                                                                                \
    Derived func() const { return Derived(intrinsic(m)); }
        AKR_F32X4_FUNC(floor_, _mm_floor_ps)
        AKR_F32X4_FUNC(ceil_, _mm_ceil_ps)
#undef AKR_F32X4_FUNC
        using Mask = typename ArrayMask<Derived>::type;
#define AKR_F32X4_FUNC(func, intrinsic)                                                                                \
    Mask func(const Derived &rhs) const { return Mask(_mm_cmp_ps(m, rhs._this().m, intrinsic)); }
        AKR_F32X4_FUNC(lt, _CMP_LT_OS)
        AKR_F32X4_FUNC(gt, _CMP_GT_OS)
        AKR_F32X4_FUNC(le, _CMP_LE_OS)
        AKR_F32X4_FUNC(ge, _CMP_GE_OS)
        AKR_F32X4_FUNC(ne, _CMP_NEQ_OS)
        AKR_F32X4_FUNC(eq, _CMP_EQ_OS)
#undef AKR_F32X4_FUNC
        static Derived select(const Mask &m, const Derived &x, const Derived &y) {
            return Derived(_mm_blendv_ps(y._this().m, x._this().m, m.m));
        }
    };

    template <typename Derived> struct i32x4 {
        union {
            __m128 m;
            __m128i mi;
            int arr[4];
        };
        const i32x4 &_this() const { return static_cast<const i32x4 &>(*this); }
        i32x4() = default;
        i32x4(__m128 m) : m(m) {}
        i32x4(__m128i mi) : mi(mi) {}
        i32x4(float v) : m(_mm_set_ps1(v)) {}
        i32x4(int i) : mi(_mm_set1_epi32(i)) {}
#define AKR_I32X4_FUNC(func, intrinsic)                                                                                \
    Derived func(const Derived &rhs) const { return Derived(intrinsic(m, rhs._this().m)); }
        AKR_I32X4_FUNC(and_, _mm_and_ps)
        AKR_I32X4_FUNC(or_, _mm_or_ps)
        AKR_I32X4_FUNC(xor_, _mm_xor_ps)
#undef AKR_I32X4_FUNC

#define AKR_I32X4_FUNC(func, intrinsic)                                                                                \
    Derived func(const Derived &rhs) const { return Derived(intrinsic(mi, rhs._this().mi)); }
        AKR_I32X4_FUNC(add, _mm_add_epi32)
        AKR_I32X4_FUNC(sub, _mm_sub_epi32)
        AKR_I32X4_FUNC(mul, _mm_mul_epi32)
        AKR_I32X4_FUNC(min_, _mm_min_epi32)
        AKR_I32X4_FUNC(max_, _mm_max_epi32)
#define AKR_I32x4_SCALAR_FUNC(func, op)                                                                                \
    Derived func(const Derived &rhs) const {                                                                           \
        i32x4 r;                                                                                                       \
        for (int i = 0; i < 4; i++) {                                                                                  \
            r.arr[i] = arr[i] op rhs._this().arr[i];                                                                   \
        }                                                                                                              \
        return Derived(r);                                                                                             \
    }
        AKR_I32x4_SCALAR_FUNC(div, /) AKR_I32x4_SCALAR_FUNC(mod, %)
#undef AKR_I32X4_FUNC
            static Derived select(const Derived &m, const Derived &x, const Derived &y) {
            return Derived(_mm_blendv_ps(y._this().m, x._this().m, m.m));
        }
    };

    template <typename Derived> struct ArrayImpl<float, 4, false, Derived> : f32x4<Derived> {
        using f32x4<Derived>::f32x4;
    };
    template <typename Derived> struct ArrayImpl<float, 3, false, Derived> : f32x4<Derived> {
        using f32x4<Derived>::f32x4;
    };
    template <typename Derived> struct ArrayImpl<float, 4, true, Derived> : maskf32x4<Derived> {
        using maskf32x4<Derived>::maskf32x4;
    };
    template <typename Derived> struct ArrayImpl<float, 3, true, Derived> : maskf32x4<Derived> {
        using maskf32x4<Derived>::maskf32x4;
    };

    template <typename Derived> struct ArrayImpl<int, 4, false, Derived> : i32x4<Derived> {
        using i32x4<Derived>::i32x4;
    };
    template <typename Derived> struct ArrayImpl<int, 3, false, Derived> : i32x4<Derived> {
        using i32x4<Derived>::i32x4;
    };
    template <typename Derived> struct ArrayImpl<int, 4, true, Derived> : i32x4<Derived> {
        using i32x4<Derived>::i32x4;
    };
    template <typename Derived> struct ArrayImpl<int, 3, true, Derived> : maskf32x4<Derived> {
        using i32x4<Derived>::i32x4;
    };

    template <typename V, int N> struct GetArrayPartition {};

    template <typename T, int N, bool isMask, int M> struct GetArrayPartition<Array<T, N, isMask>, M> {
        using type = Array<T, M, isMask>;
    };

    template <typename T, int N> constexpr int get_vector_recursive_head() {
        static_assert(!vector_recursive_is_base_case<T, N>());
        if constexpr (N > 16)
            return 16;
        else if constexpr (N > 8)
            return 8;
        else if constexpr (N > 4)
            return 4;
    }
    template <typename T, int N, bool isMask, typename Derived>
    struct ArrayImpl<T, N, isMask, Derived, std::enable_if_t<!vector_recursive_is_base_case<T, N>()>> {
        static constexpr int head = get_vector_recursive_head<T, N>();
        using Derived1 = typename GetArrayPartition<Derived, head>::type;
        using Derived2 = typename GetArrayPartition<Derived, N - head>::type;
        using Array1 = ArrayImpl<T, head, isMask, Derived1>;
        using Array2 = ArrayImpl<T, N - head, isMask, Derived2>;
        Array1 arr1;
        Array2 arr2;
        ArrayImpl() = default;
        ArrayImpl(const T &v) : arr1(v), arr2(v) {}
        ArrayImpl(const Derived1 &arr1, const Derived2 &arr2) : arr1(arr1), arr2(arr2) {}
#define AKR_VEC_FORWARD0(func)                                                                                         \
    template <typename R = Derived> std::enable_if_t<!isMask, R> func() const {                                        \
        return Derived(arr1.func(), arr2.func());                                                                      \
    }
        AKR_VEC_FORWARD0(floor_)
        AKR_VEC_FORWARD0(ceil_)
#define AKR_VEC_FORWARD1(func)                                                                                         \
    template <typename R = Derived> std::enable_if_t<!isMask, R> func(const Derived &rhs) const {                      \
        auto &r = static_cast<const ArrayImpl &>(rhs);                                                                 \
        return Derived(arr1.func(r.arr1), arr2.func(r.arr2));                                                          \
    }
        AKR_VEC_FORWARD1(add)
        AKR_VEC_FORWARD1(sub)
        AKR_VEC_FORWARD1(mul)
        AKR_VEC_FORWARD1(div)
        AKR_VEC_FORWARD1(and_)
        AKR_VEC_FORWARD1(or_)
        AKR_VEC_FORWARD1(xor_)
        AKR_VEC_FORWARD1(min_)
        AKR_VEC_FORWARD1(max_)

        using Mask = typename ArrayMask<Derived>::type;
#define AKR_VEC_FORWARD_CMP(func)                                                                                      \
    template <typename R = Mask> std::enable_if_t<!isMask, R> func(const Derived &rhs) const {                         \
        auto &r = static_cast<const ArrayImpl &>(rhs);                                                                 \
        return Mask(arr1.func(r.arr1), arr2.func(r.arr2));                                                             \
    }

        AKR_VEC_FORWARD_CMP(lt)
        AKR_VEC_FORWARD_CMP(gt)
        AKR_VEC_FORWARD_CMP(le)
        AKR_VEC_FORWARD_CMP(ge)
        AKR_VEC_FORWARD_CMP(ne)
        AKR_VEC_FORWARD_CMP(eq)

        static Derived select(const Mask &m, const Derived &x, const Derived &y) {
            return Derived(Derived1::select(m.arr1, x.arr1, y.arr1), Derived2::select(m.arr2, x.arr2, y.arr2));
        }
    };
    template <typename T, int N, bool isMask> struct Array : ArrayImpl<T, N, isMask, Array<T, N, isMask>> {
        using Base = ArrayImpl<T, N, isMask, Array<T, N, isMask>>;
        // using Mask = Array<T, N, true>;
        using ArrayImpl<T, N, isMask, Array<T, N, isMask>>::ArrayImpl;
        Array(const Base &b) : Base(b) {}
        T& operator [](int i){return reinterpret_cast<T*>(this)[i];}
        const T& operator [](int i)const{return reinterpret_cast<T*>(this)[i];}
#define GEN_ACCESS(name, idx)                                                                                          \
    const T &name() const {                                                                                            \
        static_assert(N > idx);                                                                                        \
        return (*this)[idx];                                                                                           \
    }                                                                                                                  \
    T &name() {                                                                                                        \
        static_assert(N > idx);                                                                                        \
        return (*this)[idx];                                                                                           \
    }
        GEN_ACCESS(x, 0)
        GEN_ACCESS(y, 1)
        GEN_ACCESS(z, 2)
        GEN_ACCESS(w, 3)
#define GEN_ASSIGN_OP(op, op2)                                                                                         \
    template <typename U> Array &operator op2(const U &rhs) {                                                          \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }
        GEN_ASSIGN_OP(+, +=)
        GEN_ASSIGN_OP(-, -=)
        GEN_ASSIGN_OP(*, *=)
        GEN_ASSIGN_OP(/, /=)
        GEN_ASSIGN_OP(&, &=)
        GEN_ASSIGN_OP(|, |=)
        GEN_ASSIGN_OP(^, ^=)
    };

    template <typename T, int N> using Mask = Array<T, N, true>;

    template <typename T> struct is_array : std::false_type {};
    template <typename T, int N> struct is_array<Array<T, N>> : std::true_type {};
    template <typename T> constexpr static bool is_array_v = is_array<T>::value;
    template <typename T> struct scalar_ { using type = T; };
    template <typename T, int N> struct scalar_<Array<T, N>> { using type = T; };
    template <typename T> using scalar_t = typename scalar_<T>::type;
    template <typename T, typename U> struct replace_scalar { using type = U; };
    template <typename T, int N, typename U> struct replace_scalar<Array<T, N>, U> { using type = Array<U, N>; };
    template <typename T, typename U> using replace_scalar_t = typename replace_scalar<T, U>::type;

#define AKR_ARR_BINOP(func, op)                                                                                        \
    template <typename T, int N, bool isMask, typename A = Array<T, N, isMask>>                                        \
    inline A operator op(const Array<T, N, isMask> &lhs, const Array<T, N, isMask> &rhs) {                             \
        return lhs.func(rhs);                                                                                          \
    }                                                                                                                  \
    template <typename T, int N, bool isMask, typename A = Array<T, N, isMask>, typename M = typename A::Mask>         \
    inline A operator op(const T &lhs, const Array<T, N, isMask> &rhs) {                                               \
        return A(lhs).func(rhs);                                                                                       \
    }                                                                                                                  \
    template <typename T, int N, bool isMask, typename A = Array<T, N, isMask>, typename M = typename A::Mask>         \
    inline A operator op(const Array<T, N, isMask> &lhs, const T &rhs) {                                               \
        return lhs.func(A(rhs));                                                                                       \
    }
#define AKR_ARR_BINOP_CMP(func, op)                                                                                    \
    template <typename T, int N, bool isMask, typename A = Array<T, N, isMask>, typename M = typename A::Mask>         \
    inline M operator op(const Array<T, N, isMask> &lhs, const Array<T, N, isMask> &rhs) {                             \
        return lhs.func(rhs);                                                                                          \
    }                                                                                                                  \
    template <typename T, int N, bool isMask, typename A = Array<T, N, isMask>, typename M = typename A::Mask>         \
    inline M operator op(const T &lhs, const Array<T, N, isMask> &rhs) {                                               \
        return A(lhs).func(rhs);                                                                                       \
    }                                                                                                                  \
    template <typename T, int N, bool isMask, typename A = Array<T, N, isMask>, typename M = typename A::Mask>         \
    inline M operator op(const Array<T, N, isMask> &lhs, const T &rhs) {                                               \
        return lhs.func(A(rhs));                                                                                       \
    }
    AKR_ARR_BINOP(add, +)
    AKR_ARR_BINOP(sub, -)
    AKR_ARR_BINOP(mul, *)
    AKR_ARR_BINOP(div, /)
    AKR_ARR_BINOP(and_, &)
    AKR_ARR_BINOP(or_, |)
    AKR_ARR_BINOP(xor_, ^)
    AKR_ARR_BINOP_CMP(lt, <)
    AKR_ARR_BINOP_CMP(gt, >)
    AKR_ARR_BINOP_CMP(le, <=)
    AKR_ARR_BINOP_CMP(ge, >=)
    AKR_ARR_BINOP_CMP(ne, !=)
    AKR_ARR_BINOP_CMP(eq, ==)

    template <typename T, int N, bool isMask, typename A = Array<T, N, isMask>, typename M = typename A::Mask>
    inline A select(const Mask<T, N> &m, const Array<T, N, isMask> &x, const A &y) {
        return A::select(m, x, y);
    }
    extern Array<float, 16> add_f32x12(const Array<float, 16> &a, const Array<float, 16> &b) { return a + 1.0f; }
    extern Array<int, 16> add_i32x16(const Array<int, 16> &a, const Array<int, 16> &b) { return a & b; }
} // namespace akari

int main() {
    using namespace akari;
    Array<float, 6> a(10), b(20);
    Array<float, 6> c;
    for (int i = 0; i < 6; i++) {
        c[i] = float(3 * i + 10);
    }
    for (int i = 0; i < 6; i++) {
        printf("%f\n", c[i]);
    }
    auto m = c <= 14.0f;
    for (int i = 0; i < 6; i++) {
        printf("%f\n", m[i]);
    }
    a = select(m, a, b);
    for (int i = 0; i < 6; i++) {
        printf("%f\n", a[i]);
    }
}