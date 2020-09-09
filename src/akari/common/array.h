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

#pragma once
#include <type_traits>
#include <cmath>
#include <akari/common/fwd.h>
#include <akari/common/panic.h>
#include <algorithm>
#include <cstring>

namespace akari {
    inline float select(bool c, float a, float b) { return c ? a : b; }
    inline int select(bool c, int a, int b) { return c ? a : b; }
    inline double select(bool c, double a, double b) { return c ? a : b; }
    inline bool any(bool a) { return a; }
    inline bool any(float a) { return a; }
    inline bool any(double a) { return a; }
    inline bool all(bool a) { return a; }
    inline bool all(float a) { return a; }
    inline bool all(double a) { return a; }
    template <typename T, int N, int packed>
    struct alignas(compute_align<T, N, packed>()) Array {
        static constexpr int padded_size = (int)compute_padded_size<T, N, packed>();
        T _s[padded_size] = {};
        using value_t = T;
        const T &operator[](int i) const { return _s[i]; }
        T &operator[](int i) { return _s[i]; }
        Array() = default;
        Array(const T &x) {
            for (int i = 0; i < padded_size; i++) {
                _s[i] = x;
            }
        }
        template <typename U, int P>
        explicit Array(const Array<U, N, P> &rhs) {
            if (!P) {
                for (int i = 0; i < padded_size; i++) {
                    _s[i] = T(rhs[i]);
                }
            } else {
                for (int i = 0; i < N; i++) {
                    _s[i] = T(rhs[i]);
                }
            }
        }
        Array(const T &xx, const T &yy) {
            x() = xx;
            y() = yy;
        }
        Array(const T &xx, const T &yy, const T &zz) {
            x() = xx;
            y() = yy;
            z() = zz;
        }
        Array(const T &xx, const T &yy, const T &zz, const T &ww) {
            x() = xx;
            y() = yy;
            z() = zz;
            w() = ww;
        }
#define GEN_ACCESSOR(name, idx)                                                                                        \
    const T &name() const {                                                                                            \
        static_assert(N > idx);                                                                                        \
        return _s[idx];                                                                                                \
    }                                                                                                                  \
    T &name() {                                                                                                        \
        static_assert(N > idx);                                                                                        \
        return _s[idx];                                                                                                \
    }
        GEN_ACCESSOR(x, 0)
        GEN_ACCESSOR(y, 1)
        GEN_ACCESSOR(z, 2)
        GEN_ACCESSOR(w, 3)
        GEN_ACCESSOR(r, 0)
        GEN_ACCESSOR(g, 1)
        GEN_ACCESSOR(b, 2)
        GEN_ACCESSOR(a, 3)
#undef GEN_ACCESSOR

#define GEN_ARITH_OP(op, assign_op)                                                                                    \
    Array operator op(const Array &rhs) const {                                                                        \
        Array self;                                                                                                    \
        for (int i = 0; i < padded_size; i++) {                                                                        \
            self[i] = (*this)[i] op rhs[i];                                                                            \
        }                                                                                                              \
        return self;                                                                                                   \
    }                                                                                                                  \
    Array operator op(const T &rhs) const {                                                                            \
        Array self;                                                                                                    \
        for (int i = 0; i < padded_size; i++) {                                                                        \
            self[i] = (*this)[i] op rhs;                                                                               \
        }                                                                                                              \
        return self;                                                                                                   \
    }                                                                                                                  \
    Array &operator assign_op(const Array &rhs) {                                                                      \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }
        GEN_ARITH_OP(+, +=)
        GEN_ARITH_OP(-, -=)
        GEN_ARITH_OP(*, *=)
        GEN_ARITH_OP(/, /=)
        GEN_ARITH_OP(%, %=)
        GEN_ARITH_OP(&, &=)
        GEN_ARITH_OP(|, |=)
        GEN_ARITH_OP(^, ^=)
#undef GEN_ARITH_OP
#define GEN_CMP_OP(op)                                                                                                 \
    Array<bool, N> operator op(const Array &rhs) const {                                                               \
        Array<bool, N> r;                                                                                              \
        for (int i = 0; i < N; i++) {                                                                                  \
            r[i] = (*this)[i] op rhs[i];                                                                               \
        }                                                                                                              \
        return r;                                                                                                      \
    }
        GEN_CMP_OP(==)
        GEN_CMP_OP(!=)
        GEN_CMP_OP(<=)
        GEN_CMP_OP(>=)
        GEN_CMP_OP(<)
        GEN_CMP_OP(>)
#undef GEN_CMP_OP
        friend Array operator+(const T &v, const Array &rhs) { return Array(v) + rhs; }
        friend Array operator-(const T &v, const Array &rhs) { return Array(v) - rhs; }
        friend Array operator*(const T &v, const Array &rhs) { return Array(v) * rhs; }
        friend Array operator/(const T &v, const Array &rhs) { return Array(v) / rhs; }
        Array operator-() const {
            Array self;
            for (int i = 0; i < padded_size; i++) {
                self[i] = -(*this)[i];
            }
            return self;
        }
    }; // namespace detail
    template <typename T, int N, int P>
    T dot(const Array<T, N, P> &a1, const Array<T, N, P> &a2) {
        T s = a1[0] * a2[0];
        for (int i = 1; i < N; i++) {
            s += a1[i] * a2[i];
        }
        return s;
    }
    template <typename T, int N, int P, class F>
    T reduce(const Array<T, N, P> &a, F &&f) {
        T acc = a[0];
        for (int i = 1; i < N; i++) {
            acc = f(acc, a[i]);
        }
        return acc;
    }
    template <typename T, int N, int P>
    T hsum(const Array<T, N, P> &a) {
        return reduce(a, [](const T &acc, const T &b) { return acc + b; });
    }
    template <typename T, int N, int P>
    T hprod(const Array<T, N, P> &a) {
        return reduce(a, [](const T &acc, const T &b) { return acc * b; });
    }
    template <typename T, int N, int P>
    T hmin(const Array<T, N, P> &a) {
        return reduce(a, [](const T &acc, const T &b) { return min(acc, b); });
    }
    template <typename T, int N, int P>
    T hmax(const Array<T, N, P> &a) {
        return reduce(a, [](const T &acc, const T &b) { return max(acc, b); });
    }
    template <typename T, int N, int P>
    bool any(const Array<T, N, P> &a) {
        return reduce(a, [](const T &acc, const T &b) { return acc || any(b); });
    }
    template <typename T, int N, int P>
    bool all(const Array<T, N, P> &a) {
        return reduce(a, [](const T &acc, const T &b) { return acc && all(b); });
    }
    template <typename T, int N, int P>
    Array<T, N, P> clamp(const Array<T, N, P> &x, const Array<T, N, P> &lo, const Array<T, N, P> &hi) {
        return max(min(x, hi), lo);
    }
    template <typename T, int N>
    Array<T, N> select(const Array<bool, N> &x, const Array<T, N> &a, const Array<T, N> &b) {
        Array<T, N> r;
        for (int i = 0; i < N; i++) {
            r[i] = select(x[i], a[i], b[i]);
        }
        return r;
    }
    template <typename T, int N>
    using PackedArray = Array<T, N, 1>;

#pragma GCC diagnostic push
#if defined(__GNUG__) && !defined(__clang__)
#    pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
    template <typename T, typename = std::enable_if_t<is_array_v<T>>>
    T load(const void *p) {
        T v;
        std::memcpy(&v, p, sizeof(T));
        return v;
    }
    template <typename T, int N, int P>
    void store(void *p, const Array<T, N, P> &a) {
        std::memcpy(p, &a, sizeof(Array<T, N, P>));
    }
#pragma GCC diagnostic pop
    template <typename T, int N>
    T length(const Array<T, N> &a) {
        return sqrt(dot(a, a));
    }
    template <typename T, int N>
    Array<T, N> normalize(const Array<T, N> &a) {
        return a / sqrt(dot(a, a));
    }
    template <int... args, typename T, int N>
    auto shuffle(const Array<T, N> &a) {
        constexpr int pack[] = {args...};
        static_assert(((args < N) && ...));
        Array<T, sizeof...(args)> s;
        for (size_t i = 0; i < sizeof...(args); i++) {
            s[i] = a[pack[i]];
        }
        return s;
    }
#define FWD_MATH_FUNC1(name)                                                                                           \
                                                                                                                       \
    template <typename V, int N, int P>                                                                                \
    Array<V, N, P> _##name(const Array<V, N, P> &v) {                                                                  \
        Array<V, N, P> ans;                                                                                            \
        using std::name;                                                                                               \
        for (int i = 0; i < N; i++) {                                                                                  \
            ans[i] = name(v[i]);                                                                                       \
        }                                                                                                              \
        return ans;                                                                                                    \
    }                                                                                                                  \
    template <typename V, int N, int P>                                                                                \
    Array<V, N, P> name(const Array<V, N, P> &v) {                                                                     \
        return _##name(v);                                                                                             \
    }
#define FWD_MATH_FUNC2(name)                                                                                           \
                                                                                                                       \
    template <typename V, int N, int P>                                                                                \
    Array<V, N, P> _##name(const Array<V, N, P> &v1, const Array<V, N, P> &v2) {                                       \
        Array<V, N, P> ans;                                                                                            \
        using std::name;                                                                                               \
        for (int i = 0; i < N; i++) {                                                                                  \
            ans[i] = name(v1[i], v2[i]);                                                                               \
        }                                                                                                              \
        return ans;                                                                                                    \
    }                                                                                                                  \
    template <typename V, int N, int P>                                                                                \
    Array<V, N, P> _##name(const V &v1, const Array<V, N, P> &v2) {                                                    \
        Array<V, N, P> ans;                                                                                            \
        using std::name;                                                                                               \
        for (int i = 0; i < N; i++) {                                                                                  \
            ans[i] = name(v1, v2[i]);                                                                                  \
        }                                                                                                              \
        return ans;                                                                                                    \
    }                                                                                                                  \
    template <typename V, int N, int P>                                                                                \
    Array<V, N, P> _##name(const Array<V, N, P> &v1, const V &v2) {                                                    \
        Array<V, N, P> ans;                                                                                            \
        using std::name;                                                                                               \
        for (int i = 0; i < N; i++) {                                                                                  \
            ans[i] = name(v1[i], v2);                                                                                  \
        }                                                                                                              \
        return ans;                                                                                                    \
    }                                                                                                                  \
    template <typename V, int N, int P>                                                                                \
    Array<V, N, P> name(const Array<V, N, P> &v1, const Array<V, N, P> &v2) {                                          \
        return _##name(v1, v2);                                                                                        \
    }                                                                                                                  \
    template <typename V, int N, int P>                                                                                \
    Array<V, N, P> name(const V &v1, const Array<V, N, P> &v2) {                                                       \
        return _##name(v1, v2);                                                                                        \
    }                                                                                                                  \
    template <typename V, int N, int P>                                                                                \
    Array<V, N, P> name(const Array<V, N, P> &v1, const V &v2) {                                                       \
        return _##name(v1, v2);                                                                                        \
    }
    FWD_MATH_FUNC1(floor)
    FWD_MATH_FUNC1(ceil)
    FWD_MATH_FUNC1(abs)
    FWD_MATH_FUNC1(log)
    FWD_MATH_FUNC1(sin)
    FWD_MATH_FUNC1(cos)
    FWD_MATH_FUNC1(tan)
    FWD_MATH_FUNC1(exp)
    FWD_MATH_FUNC1(sqrt)
    FWD_MATH_FUNC1(asin)
    FWD_MATH_FUNC1(acos)
    FWD_MATH_FUNC1(atan)
    FWD_MATH_FUNC2(atan2)
    FWD_MATH_FUNC2(pow)
    FWD_MATH_FUNC2(min)
    FWD_MATH_FUNC2(max)
#undef FWD_MATH_FUNC1
#undef FWD_MATH_FUNC2
#define AKR_ARRAY_IMPORT_ARITH_OP(op, assign_op, Base, Self)                                                           \
    Self operator op(const Self &rhs) const {                                                                          \
        return Self(static_cast<const Base &>(*this) op static_cast<const Base &>(rhs));                               \
    }                                                                                                                  \
    Self operator op(const Self::value_t &rhs) const { return Self(static_cast<const Base &>(*this) op rhs); }         \
    Self operator assign_op(const Self &rhs) {                                                                         \
        *this = Self(static_cast<Base &>(*this) op static_cast<const Base &>(rhs));                                    \
        return *this;                                                                                                  \
    }
#define AKR_ARRAY_IMPORT(Base, Self)                                                                                   \
    AKR_ARRAY_IMPORT_ARITH_OP(+, +=, Base, Self)                                                                       \
    AKR_ARRAY_IMPORT_ARITH_OP(-, -=, Base, Self)                                                                       \
    AKR_ARRAY_IMPORT_ARITH_OP(*, *=, Base, Self)                                                                       \
    AKR_ARRAY_IMPORT_ARITH_OP(/, /=, Base, Self)                                                                       \
    AKR_ARRAY_IMPORT_ARITH_OP(%, %=, Base, Self)                                                                       \
    friend Self operator+(const Self::value_t &v, const Self &rhs) { return Self(v) + rhs; }                           \
    friend Self operator-(const Self::value_t &v, const Self &rhs) { return Self(v) - rhs; }                           \
    friend Self operator*(const Self::value_t &v, const Self &rhs) { return Self(v) * rhs; }                           \
    friend Self operator/(const Self::value_t &v, const Self &rhs) { return Self(v) / rhs; }                           \
    Self operator-() const { return Self(-static_cast<const Base &>(*this)); }                                         \
    Self &operator=(const Base &base) {                                                                                \
        static_cast<Base &>(*this) = base;                                                                             \
        return *this;                                                                                                  \
    }                                                                                                                  \
    Self(const Base &base) : Base(base) {}
} // namespace akari