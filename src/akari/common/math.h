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
    template <typename Float> struct Constants {
        static constexpr Float Inf = std::numeric_limits<Float>::infinity();
        static constexpr Float Pi = 3.1415926535897932384f;
        static constexpr Float InvPi = 1.0f / Pi;
    };
    inline bool select(bool c, float a, float b) { return c ? a : b; }
    inline bool select(bool c, int a, int b) { return c ? a : b; }
    inline bool select(bool c, double a, double b) { return c ? a : b; }
    inline bool any(bool a) { return a; }
    inline bool all(bool a) { return a; }
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
        for (int i = 0; i < padded_size; i++) {                                                                        \
            self[i] = (*this)[i] op rhs[i];                                                                            \
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
        for (int i = 0; i < padded_size; i++) {                                                                        \
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
    template <typename T, int N> T dot(const Array<T, N> &a1, const Array<T, N> &a2) {
        T s = a1[0] * a2[0];
        for (int i = 1; i < N; i++) {
            s += a1[i] * a2[i];
        }
        return s;
    }
    template <typename T, int N, class F> T reduce(const Array<T, N> &a, F &&f) {
        T acc = a[0];
        for (int i = 1; i < N; i++) {
            acc = f(acc, a[i]);
        }
        return acc;
    }
    template <typename T, int N> T hsum(const Array<T, N> &a) {
        return reduce(a, [](T acc, T b) { return acc + b; });
    }
    template <typename T, int N> T hprod(const Array<T, N> &a) {
        return reduce(a, [](T acc, T b) { return acc * b; });
    }
    template <typename T, int N> T hmin(const Array<T, N> &a) {
        return reduce(a, [](T acc, T b) { return min(acc, b); });
    }
    template <typename T, int N> T hmax(const Array<T, N> &a) {
        return reduce(a, [](T acc, T b) { return max(acc, b); });
    }
    template <typename T, int N> bool any(const Array<T, N> &a) {
        return reduce(a, [](T acc, T b) { return acc || any(b); });
    }
    template <typename T, int N> bool all(const Array<T, N> &a) {
        return reduce(a, [](T acc, T b) { return acc && all(b); });
    }
    template <typename T, int N> Array<T, N> clamp(const Array<T, N> &x, const Array<T, N> &lo, const Array<T, N> &hi) {
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
    template <typename T, typename = std::enable_if_t<is_array_v<T>>> T load(const value_t<T> *p) {
        T v;
        std::memcpy(&v, p, sizeof(T));
        return v;
    }
    template <typename T, int N> T length(const Array<T, N> &a) { return sqrt(dot(a, a)); }
    template <typename T, int N> Array<T, N> normalize(const Array<T, N> &a) { return a / sqrt(dot(a, a)); }
    template <int... args, typename T, int N> auto shuffle(const Array<T, N> &a) {
        constexpr int pack[] = {args...};
        static_assert(((args < N) && ...));
        Array<T, sizeof...(args)> s;
        for (int i = 0; i < sizeof...(args); i++) {
            s[i] = a[pack[i]];
        }
        return s;
    }
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
    FWD_MATH_FUNC2(min)
    FWD_MATH_FUNC2(max)
    template <typename V, int N, bool P> Array<V, N, P> pow(const Array<V, N, P> &v1, V p) {
        return pow(v1, Array<V, N, P>(p));
    }
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
    Self operator-() const { return Self(-static_cast<const Base &>(*this)); }                                         \
    Self &operator=(const Base &base) {                                                                                \
        static_cast<Base &>(*this) = base;                                                                             \
        return *this;                                                                                                  \
    }                                                                                                                  \
    Self(const Base &base) : Base(base) {}

    template <typename Value, int N> struct Vector : Array<Value, N> {
        using Base = Array<Value, N>;
        using Base::Base;
        using value_t = Value;
        static constexpr size_t size = N;
        AKR_ARRAY_IMPORT(Base, Vector)
    };
    template <typename Value, int N> struct Point : Array<Value, N> {
        using Base = Array<Value, N>;
        using Base::Base;
        using value_t = Value;
        static constexpr size_t size = N;
        AKR_ARRAY_IMPORT(Base, Point)
    };
    template <typename Value, int N> struct Normal : Array<Value, N> {
        using Base = Array<Value, N>;
        using Base::Base;
        using value_t = Value;
        static constexpr size_t size = N;
        AKR_ARRAY_IMPORT(Base, Normal)
    };

    // template <typename Value, int N> Vector<Value, N> operator-(const Point<Value, N> &p1, const Point<Value, N> &p2)
    // {
    //     Vector<Value, N> v;
    //     for (int i = 0; i < N; i++) {
    //         v[i] = p1[i] - p2[i];
    //     }
    //     return v;
    // }
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
        return (1.0f - uv.x() - uv.y()) * v0 + uv.x() * v1 + uv.y() * v2;
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
            static_assert(sizeof(Float) * N * N == sizeof(rows));
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
        Matrix operator+(const Matrix &rhs) const { return Matrix(rows + rhs.rows); }
        Matrix operator-(const Matrix &rhs) const { return Matrix(rows - rhs.rows); }
        Array<Float, N> operator*(const Array<Float, N> &v) const {
            Array<Float, N> r;
            for (int i = 0; i < N; i++) {
                r[i] = dot(row(i), v);
            }
            return r;
        }
        Matrix operator*(const Matrix &rhs) const {
            Matrix m;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    m(i, j) = dot(row(i), rhs.col(j));
                }
            }
            return m;
        }
        Matrix transpose() const {
            Matrix m;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    m(i, j) = (*this)(j, i);
                }
            }
            return m;
        }
        Matrix inverse() const {
            // from pbrt
            auto &m = *this;
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

    template <typename T> Vector<T, 3> cross(const Vector<T, 3> &v1, const Vector<T, 3> &v2) {
        T v1x = v1.x(), v1y = v1.y(), v1z = v1.z();
        T v2x = v2.x(), v2y = v2.y(), v2z = v2.z();
        return Vector<T, 3>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z), (v1x * v2y) - (v1y * v2x));
    }
    template <typename T> Normal<T, 3> cross(const Normal<T, 3> &v1, const Normal<T, 3> &v2) {
        T v1x = v1.x(), v1y = v1.y(), v1z = v1.z();
        T v2x = v2.x(), v2y = v2.y(), v2z = v2.z();
        return Normal<T, 3>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z), (v1x * v2y) - (v1y * v2x));
    }
    AKR_VARIANT struct Ray {
        // Float time = 0.0f;
        AKR_IMPORT_CORE_TYPES()
        Point3f o;
        Vector3f d;
        Float tmin, tmax;
        Ray(const Point3f &o, const Vector3f &d, Float tmin, Float tmax = = std::numeric_limits<Float>::infinity())
            : o(o), d(d), tmin(tmin), tmax(tmax) {}
        Point3f operator()(Float t) const { return o + t * d; }
    };

    template <typename Vector> struct Frame {
        using Value = typename Vector::value_t;
        using Vector3f = Vector;
        using Normal3f = Normal<Value, 3>;
        Frame() = default;
        static inline void compute_local_frame(const Normal3f &v1, Vector3f *v2, Vector3f *v3) {
            if (std::abs(v1.x()) > std::abs(v1.y()))
                *v2 = Vector3f(-v1.z(), (0), v1.x()) / std::sqrt(v1.x() * v1.x() + v1.z() * v1.z());
            else
                *v2 = Vector3f((0), v1.z(), -v1.y()) / std::sqrt(v1.y() * v1.y() + v1.z() * v1.z());
            *v3 = normalize(cross(Vector3f(v1), *v2));
        }
        explicit Frame(const Normal3f &v) : normal(v) { compute_local_frame(v, &T, &B); }

        [[nodiscard]] Vector3f world_to_local(const Vector3f &v) const {
            return Vector3f(dot(T, v), dot(normal, v), dot(B, v));
        }

        [[nodiscard]] Vector3f local_to_world(const Vector3f &v) const {
            return Vector3f(v.x() * T + v.y() * Vector3f(normal) + v.z() * B);
        }

        Normal3f normal;
        Vector3f T, B;
    };

    template <typename Float> struct Transform {
        AKR_IMPORT_CORE_TYPES()
        Matrix4f m, minv;
        Matrix3f m3, m3inv;
        Transform() = default;
        Transform(const Matrix4f &m) : Transform(m, m.inverse()) {}
        Transform(const Matrix4f &m, const Matrix4f &minv) : m(m), minv(minv) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    m3(i, j) = m(i, j);
                }
            }
            m3inv = m3.inverse();
        }
        Transform inverse() const { return Transform(minv, m); }
        Point3f operator*(const Point3f &p) const {
            Vector4f v(p.x(), p.y(), p.z(), 1.0);
            v = m * v;
            Point3f q(v.x(), v.y(), v.z());
            if (v.w() != 1.0) {
                q /= v.w();
            }
            return q;
        }
        Transform operator*(const Transform &t) const { return Transform(m * t.m); }
        Vector3f operator*(const Vector3f &v) const { return m3 * v; }
        Normal3f operator*(const Normal3f &n) const { return m3inv.transpose() * n; }
        template <typename Spectrum> Ray<Float, Spectrum> operator*(const Ray<Float, Spectrum> &ray) const {
            using Ray3f = Ray<Float, Spectrum>;
            auto &T = *this;
            auto d2 = T * ray.d;
            auto len1 = dot(ray.d, ray.d);
            auto len2 = dot(d2, d2);
            auto scale = len2 / len1;
            return Ray3f(T * ray.o, d2, ray.tmin * scale, ray.tmax * scale);
        }

        static Transform translate(const Vector3f &v) {
            float m[] = {1, 0, 0, v.x(), 0, 1, 0, v.y(), 0, 0, 1, v.z(), 0, 0, 0, 1};
            return Transform(Matrix4f(m));
        }
        static Transform scale(const Vector3f &s) {
            float m[] = {s.x(), 0, 0, 0, 0, s.y(), 0, 0, 0, 0, s.z(), 0, 0, 0, 0, 1};
            return Transform(Matrix4f(m));
        }
        static Transform rotate_x(Float theta) {
            Float sinTheta = std::sin(theta);
            Float cosTheta = std::cos(theta);
            float m[] = {1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1};
            return Transform(m, Matrix4f(m).transpose());
        }

        static Transform rotate_y(Float theta) {
            Float sinTheta = std::sin(theta);
            Float cosTheta = std::cos(theta);
            float m[] = {cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0, 0, 0, 0, 1};
            return Transform(m, Matrix4f(m).transpose());
        }

        static Transform rotate_z(Float theta) {
            Float sinTheta = std::sin(theta);
            Float cosTheta = std::cos(theta);
            float m[] = {cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            return Transform(m, Matrix4f(m).transpose());
        }
    };

    template <typename Point> struct BoundingBox {
        using Float = value_t<Point>;
        static constexpr auto N = array_size_v<Point>;
        using Vector = akari::Vector<Float, N>;
        Point pmin, pmax;
        BoundingBox() { reset(); }
        void reset() {
            pmin = Point(Constants<Float>::Inf);
            pmax = Point(-Constants<Float>::Inf);
        }
        Vector extents() const { return pmax - pmin; }
        Vector skze() const { return extents(); }
        Vector offset(const Point &p) { return (p - pmin) / extents(); }
        void expand(const Point &p) {
            pmin = min(pmin, p);
            pmax = max(pmax, p);
        }
        static BoundingBox merge(const BoundingBox &b1, const BoundingBox &b2) {
            return BoundingBox(min(b1.pmin, b2.pmin), max(p1.pmax, b2.pmax));
        }
        Point centroid() const { return extents() * 0.5f + p_min; }
        Float surface_area() const {
            if constexpr (N == 3) {
                auto ext = extents();
                return hsum(akari::shuffle<1, 2, 0>(ext) * ext) * Float(2);
            } else {
                auto ext = extents();
                Float result = Float(0);
                for (size_t i = 0; i < N; ++i) {
                    Float term = Float(1);
                    for (size_t j = 0; j < N; ++j) {
                        if (i == j)
                            continue;
                        term *= ext[j];
                    }
                    result += term;
                }
                return result * Float(2);
            }
        }
    };

    template <typename Float> inline Float degrees(Float x) { return x * Constants<value_t<Float>>::InvPi * 180.0f; }
    template <typename Float> inline Float radians(Float x) { return x * Constants<value_t<Float>>::Pi / 180.0f; }
} // namespace akari