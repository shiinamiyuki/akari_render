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
#include <akari/common/fwd.hpp>
#include <akari/common/panic.hpp>
#include <algorithm>
#include <cstring>
namespace akari {

    template <typename T, size_t N, int packed> constexpr int compute_padded_size() {
        if (!std::is_fundamental_v<T>) {
            return N;
        }
        if (packed || N <= 2) {
            return N;
        }
        if (sizeof(T) == 1) {
            // round to 128 bits
            return (N + 15u) & ~15u;
        } else if (sizeof(T) == 2) {
            // round to 128 bits
            return (N + 7u) & ~7u;
        } else if (sizeof(T) == 4) {
            // round to 128 bits
            return (N + 3u) & ~3u;
        } else if (sizeof(T) == 8) {
            // round to 128 bits
            return (N + 1u) & ~1u;
        } else {
            return N;
        }
    }
    template <typename T, size_t N, int packed> constexpr int compute_align() {
        if (!std::is_fundamental_v<T>) {
            return alignof(T);
        }
        if (packed || N <= 2) {
            return alignof(T);
        }
        return 128 / 32;
    }
    template <typename T, size_t N, int packed = 0> struct alignas(compute_align<T, N, packed>()) Array;

    template <typename T, size_t N, int packed> struct alignas(compute_align<T, N, packed>()) Array {
        static constexpr size_t padded_size = compute_padded_size<T, N, packed>();
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
        template <typename U, bool P> explicit Array(const Array<U, N, P> &rhs) {
            if (!P) {
                for (int i = 0; i < padded_size; i++) {
                    _s[i] = rhs[i];
                }
            } else {
                for (int i = 0; i < N; i++) {
                    _s[i] = rhs[i];
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
        for (int i = 0; i < N; i++) {                                                                                  \
            self[i] = (*this)[i] op rhs[i];                                                                               \
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
#undef GEN_ARITH_OP
        friend Array operator*(const T &v, const Array &rhs) { return Array(v) * rhs; }
        friend Array operator/(const T &v, const Array &rhs) { return Array(v) / rhs; }
        Array operator-() const {
            Array self;
            for (int i = 0; i < N; i++) {
                self[i] = -(*this)[i];
            }
            return self;
        }

        friend T dot(const Array &a1, const Array &a2) {
            T s = a1[0] * a2[0];
            for (int i = 1; i < N; i++) {
                s += a1[i] * a2[i];
            }
            return s;
        }
    }; // namespace detail

#define FWD_MATH_FUNC1(name)                                                                                           \
    using std::name;                                                                                                   \
    template <typename V, int N, bool P> Array<V, N, P> _##name(const Array<V, N, P> &v) {                             \
        Array<V, N, P> ans;                                                                                            \
        for (int i = 0; i < N; i++) {                                                                                  \
            ans[i] = name(v[i]);                                                                                       \
        }                                                                                                              \
        return ans;                                                                                                    \
    }                                                                                                                  \
    template <typename V, int N, bool P> Array<V, N, P> name(const Array<V, N, P> &v) { return _##name(v); }
#define FWD_MATH_FUNC2(name)                                                                                           \
    using std::name;                                                                                                   \
    template <typename V, int N, bool P> Array<V, N, P> _##name(const Array<V, N, P> &v1, const Array<V, N, P> &v2) {  \
        Array<V, N, P> ans;                                                                                            \
        for (int i = 0; i < N; i++) {                                                                                  \
            ans[i] = name(v1[i], v2[i]);                                                                               \
        }                                                                                                              \
        return ans;                                                                                                    \
    }                                                                                                                  \
    template <typename V, int N, bool P> Array<V, N, P> name(const Array<V, N, P> &v1, const Array<V, N, P> &v2) {     \
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

#undef FWD_MATH_FUNC1
#undef FWD_MATH_FUNC2
#define AKR_ARRAY_IMPORT_ARITH_OP(op, assign_op, Base, Self)                                                           \
    Self operator op(const Self &rhs) const {                                                                          \
        return Self(static_cast<const Base &>(*this) + static_cast<const Base &>(rhs));                                \
    }                                                                                                                  \
    Self operator assign_op(const Self &rhs) {                                                                         \
        *this = Self(static_cast<Base &>(*this) + static_cast<const Base &>(rhs));                                     \
        return *this;                                                                                                  \
    }
#define AKR_ARRAY_IMPORT(Base, Self)                                                                                   \
    AKR_ARRAY_IMPORT_ARITH_OP(+, +=, Base, Self)                                                                       \
    AKR_ARRAY_IMPORT_ARITH_OP(-, -=, Base, Self)                                                                       \
    AKR_ARRAY_IMPORT_ARITH_OP(*, *=, Base, Self)                                                                       \
    AKR_ARRAY_IMPORT_ARITH_OP(/, /=, Base, Self)                                                                       \
    AKR_ARRAY_IMPORT_ARITH_OP(%, %=, Base, Self)                                                                       \
    friend Self operator*(const Self::value_t &v, const Self &rhs) { return Self(v) * rhs; }                           \
    friend Array operator/(const Self::value_t &v, const Self &rhs) { return Self(v) / rhs; }                          \
    Self operator-() const { return Self(-static_cast<const Base &>(*this)); }

    template <typename Value, int N> struct Vector : Array<Value, N> {
        using Base = Array<Value, N>;
        using Base::Base;
        using value_t = Value;
        AKR_ARRAY_IMPORT(Base, Vector)
    };
    template <typename Value, int N> struct Point : Array<Value, N> {
        using Base = Array<Value, N>;
        using Base::Base;
        using value_t = Value;
        AKR_ARRAY_IMPORT(Base, Point)
    };
    template <typename Value, int N> struct Normal : Array<Value, N> {
        using Base = Array<Value, N>;
        using Base::Base;
        using value_t = Value;
        AKR_ARRAY_IMPORT(Base, Normal)
    };

    template <typename Value, int N> Vector<Value, N> operator-(const Point<Value, N> &p1, const Point<Value, N> &p2) {
        Vector<Value, N> v;
        for (int i = 0; i < N; i++) {
            v[i] = p1[i] - p2[i];
        }
        return v;
    }
    template <typename Value, int N> Point<Value, N> operator+(const Point<Value, N> &p1, const Vector<Value, N> &v) {
        Point<Value, N> p2;
        for (int i = 0; i < N; i++) {
            p2[i] = p1[i] + v[i];
        }
        return p2;
    }
    template <typename Value, int N> Point<Value, N> operator+(const Vector<Value, N> &v, const Point<Value, N> &p1) {
        Point<Value, N> p2;
        for (int i = 0; i < N; i++) {
            p2[i] = p1[i] + v[i];
        }
        return p2;
    }

    template <typename V, typename V2> inline V lerp3(const V &v0, const V &v1, const V &v2, const V2 &uv) {
        return (1.0f - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
    }

    template <typename Float, int N> struct Matrix {
        Array<Array<Float, N>, N> rows;
        Matrix(Float v = 1.0) {
            for (int i = 0; i < N; i++) {
                rows[i][i] = v;
            }
        }
        Matrix(const Array<Array<Float, N>, N> &m) : rows(m) {}
        Matrix(Float m[N][N]) { std::memcpy(&rows, m, sizeof(Float) * N * N); }
        Matrix(Float m[N * N]) {
            static_assert(sizeof(m) == sizeof(rows));
            std::memcpy(&rows, m, sizeof(Float) * N * N);
        }
        const Float &operator()(int i, int j) const { return rows[i][j]; }
        Float &operator()(int i, int j) { return rows[i][j]; }
        const Array<Float, N> &row(int i) const { return rows[i]; }
        Array<Float, N> &row(int i) { return rows[i]; }
        Array<Float, N> col(int i) const {
            Array<Float, N> r;
            for (int j = 0; j < N; j++) {
                r[j] = rows[j][i];
            }
            return r;
        }
        Matrix operator+(const Matrix &rhs) { return Matrix(rows + rhs.rows); }
        Matrix operator-(const Matrix &rhs) { return Matrix(rows - rhs.rows); }
        Matrix operator*(const Matrix &rhs) {
            Matrix m;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    m(i, j) = dot(row(i), rhs.col(j));
                }
            }
            return m;
        }
        friend Matrix inverse(const Matrix &m) {
            // from pbrt
            int indxc[N], indxr[N];
            int ipiv[N] = {0};
            Float minv[N][N];
            memcpy(minv, &m.rows, N * N * sizeof(Float));
            for (int i = 0; i < N; i++) {
                int irow = 0, icol = 0;
                Float big = 0.f;
                // Choose pivot
                for (int j = 0; j < N; j++) {
                    if (ipiv[j] != 1) {
                        for (int k = 0; k < N; k++) {
                            if (ipiv[k] == 0) {
                                if (std::abs(minv[j][k]) >= big) {
                                    big = Float(std::abs(minv[j][k]));
                                    irow = j;
                                    icol = k;
                                }
                            } else if (ipiv[k] > 1)
                                AKR_PANIC("Singular matrix in MatrixInvert");
                        }
                    }
                }
                ++ipiv[icol];
                // Swap rows _irow_ and _icol_ for pivot
                if (irow != icol) {
                    for (int k = 0; k < N; ++k)
                        std::swap(minv[irow][k], minv[icol][k]);
                }
                indxr[i] = irow;
                indxc[i] = icol;
                if (minv[icol][icol] == 0.f)
                    AKR_PANIC("Singular matrix in MatrixInvert");

                // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
                Float pivinv = 1.0 / minv[icol][icol];
                minv[icol][icol] = 1.0;
                for (int j = 0; j < N; j++)
                    minv[icol][j] *= pivinv;

                // Subtract this row from others to zero out their columns
                for (int j = 0; j < N; j++) {
                    if (j != icol) {
                        Float save = minv[j][icol];
                        minv[j][icol] = 0;
                        for (int k = 0; k < N; k++)
                            minv[j][k] -= minv[icol][k] * save;
                    }
                }
            }
            // Swap columns to reflect permutation
            for (int j = N - 1; j >= 0; j--) {
                if (indxr[j] != indxc[j]) {
                    for (int k = 0; k < N; k++)
                        std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
                }
            }
            return Matrix(minv);
        }
    };
} // namespace akari