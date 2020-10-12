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
#include <akari/common/array.h>
#include <akari/common/static-reflect.h>
#include <algorithm>
#include <cstring>

namespace akari {
    template <typename Float>
    struct Constants {
        AKR_XPU static constexpr Float Inf() { return std::numeric_limits<Float>::infinity(); }
        AKR_XPU static constexpr Float Pi() { return Float(3.1415926535897932384f); }
        AKR_XPU static constexpr Float Pi2() { return Pi() / Float(2.0f); }
        AKR_XPU static constexpr Float Pi4() { return Pi() / Float(4.0f); }
        AKR_XPU static constexpr Float InvPi() { return Float(1.0f) / Pi(); }
        AKR_XPU static constexpr Float Eps() { return Float(0.001f); }
        AKR_XPU static constexpr Float ShadowEps() { return Float(0.0001f); }
    };
    template <typename T, typename Float>
    AKR_XPU T lerp(T a, T b, Float t) {
        return a * (Float(1.0) - t) + b * t;
    }
    template <typename V, typename V2>
    AKR_XPU inline V lerp3(const V &v0, const V &v1, const V &v2, const V2 &uv) {
        return (1.0f - uv[0] - uv[1]) * v0 + uv[0] * v1 + uv[1] * v2;
    }

    template <typename Float, int N>
    struct Matrix {
        Array<Array<Float, N>, N> rows;
        AKR_XPU explicit Matrix(Float v = 1.0) {
            for (int i = 0; i < N; i++) {
                rows[i][i] = v;
            }
        }
        AKR_XPU Matrix(const Array<Array<Float, N>, N> &m) : rows(m) {}
        AKR_XPU Matrix(Float m[N][N]) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    rows[i][j] = m[i][j];
                }
            }
        }
        AKR_XPU Matrix(Float m[N * N]) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    rows[i][j] = m[i * 4 + j];
                }
            }
        }
        AKR_XPU const Float &operator()(int i, int j) const { return rows[i][j]; }
        AKR_XPU Float &operator()(int i, int j) { return rows[i][j]; }
        AKR_XPU const Array<Float, N> &row(int i) const { return rows[i]; }
        AKR_XPU Array<Float, N> &row(int i) { return rows[i]; }
        AKR_XPU Array<Float, N> col(int i) const {
            Array<Float, N> r;
            for (int j = 0; j < N; j++) {
                r[j] = rows[j][i];
            }
            return r;
        }
        AKR_XPU Matrix operator+(const Matrix &rhs) const { return Matrix(rows + rhs.rows); }
        AKR_XPU Matrix operator-(const Matrix &rhs) const { return Matrix(rows - rhs.rows); }
        AKR_XPU Array<Float, N> operator*(const Array<Float, N> &v) const {
            Array<Float, N> r;
            for (int i = 0; i < N; i++) {
                r[i] = dot(row(i), v);
            }
            return r;
        }
        AKR_XPU Matrix operator*(const Matrix &rhs) const {
            Matrix m;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    m(i, j) = dot(row(i), rhs.col(j));
                }
            }
            return m;
        }
        AKR_XPU Matrix transpose() const {
            Matrix m;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    m(i, j) = (*this)(j, i);
                }
            }
            return m;
        }
        AKR_XPU Matrix inverse() const {
            // from pbrt
            auto &m = *this;
            int indxc[N], indxr[N];
            int ipiv[N] = {0};
            auto minv = m.rows;
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

    template <typename T, int P>
    AKR_XPU Array<T, 3, P> cross(const Array<T, 3, P> &v1, const Array<T, 3, P> &v2) {
        T v1x = v1.x, v1y = v1.y, v1z = v1.z;
        T v2x = v2.x, v2y = v2.y, v2z = v2.z;
        return Array<T, 3, P>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z), (v1x * v2y) - (v1y * v2x));
    }

    AKR_VARIANT struct Ray {
        // Float time = 0.0f;
        using Float = typename C::Float;
        AKR_IMPORT_CORE_TYPES()
        Float3 o;
        Float3 d;
        Float tmin = -1, tmax = -1;
        Ray() = default;
        AKR_XPU Ray(const Float3 &o, const Float3 &d, Float tmin = Constants<Float>::Eps(),
                    Float tmax = std::numeric_limits<Float>::infinity())
            : o(o), d(d), tmin(tmin), tmax(tmax) {}
        static AKR_XPU Ray spawn_to(const Float3 &p0, const Float3 &p1) {
            Float3 dir = p1 - p0;
            return Ray(p0, dir, Constants<Float>::Eps(), Float(1.0f) - Constants<Float>::ShadowEps());
        }
        AKR_XPU Float3 operator()(Float t) const { return o + t * d; }
    };

    template <typename Vector>
    struct Frame {
        using Value = typename Vector::value_t;
        using Float3 = Vector;
        Frame() = default;
        static AKR_XPU inline void compute_local_frame(const Float3 &v1, Float3 *v2, Float3 *v3) {
            if (std::abs(v1.x) > std::abs(v1.y))
                *v2 = Float3(-v1.z, (0), v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
            else
                *v2 = Float3((0), v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
            *v3 = normalize(cross(Float3(v1), *v2));
        }
        AKR_XPU explicit Frame(const Float3 &v) : normal(v) { compute_local_frame(v, &T, &B); }

        [[nodiscard]] AKR_XPU Float3 world_to_local(const Float3 &v) const {
            return Float3(dot(T, v), dot(normal, v), dot(B, v));
        }

        [[nodiscard]] AKR_XPU Float3 local_to_world(const Float3 &v) const {
            return Float3(v.x * T + v.y * Float3(normal) + v.z * B);
        }

        Float3 normal;
        Float3 T, B;
    };

    template <typename Float>
    struct Transform {
        AKR_IMPORT_CORE_TYPES()
        Matrix4f m, minv;
        Matrix3f m3, m3inv;
        Transform() = default;
        AKR_XPU Transform(const Matrix4f &m) : Transform(m, m.inverse()) {}
        AKR_XPU Transform(const Matrix4f &m, const Matrix4f &minv) : m(m), minv(minv) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    m3(i, j) = m(i, j);
                }
            }
            m3inv = m3.inverse();
        }
        AKR_XPU Transform inverse() const { return Transform(minv, m); }
        AKR_XPU Float3 apply_point(const Float3 &p) const {
            Float4 v(p.x, p.y, p.z, 1.0);
            v = m * v;
            Float3 q(v.x, v.y, v.z);
            if (v.w != 1.0) {
                q /= v.w;
            }
            return q;
        }
        AKR_XPU Transform operator*(const Transform &t) const { return Transform(m * t.m); }
        AKR_XPU Float3 apply_vector(const Float3 &v) const { return m3 * v; }
        AKR_XPU Float3 apply_normal(const Float3 &n) const { return m3inv.transpose() * n; }
        template <typename Spectrum, typename C = Config<Float, Spectrum>>
        AKR_XPU Ray<C> apply_ray(const Ray<C> &ray) const {
            using Ray3f = Ray<C>;
            auto &T = *this;
            auto d2 = T * ray.d;
            auto len1 = dot(ray.d, ray.d);
            auto len2 = dot(d2, d2);
            auto scale = len2 / len1;
            return Ray3f(T * ray.o, d2, ray.tmin * scale, ray.tmax * scale);
        }

        static AKR_XPU Transform translate(const Float3 &v) {
            Float m[] = {1, 0, 0, v.x, 0, 1, 0, v.y, 0, 0, 1, v.z, 0, 0, 0, 1};
            return Transform(Matrix4f(m));
        }
        static AKR_XPU Transform scale(const Float3 &s) {
            Float m[] = {s.x, 0, 0, 0, 0, s.y, 0, 0, 0, 0, s.z, 0, 0, 0, 0, 1};
            return Transform(Matrix4f(m));
        }
        static AKR_XPU Transform rotate_x(Float theta) {
            Float sinTheta = sin(theta);
            Float cosTheta = cos(theta);
            Float m[] = {1, 0, 0, 0, 0, cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1};
            return Transform(m, Matrix4f(m).transpose());
        }

        static AKR_XPU Transform rotate_y(Float theta) {
            Float sinTheta = sin(theta);
            Float cosTheta = cos(theta);
            Float m[] = {cosTheta, 0, sinTheta, 0, 0, 1, 0, 0, -sinTheta, 0, cosTheta, 0, 0, 0, 0, 1};
            return Transform(m, Matrix4f(m).transpose());
        }

        static AKR_XPU Transform rotate_z(Float theta) {
            Float sinTheta = sin(theta);
            Float cosTheta = cos(theta);
            Float m[] = {cosTheta, -sinTheta, 0, 0, sinTheta, cosTheta, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            return Transform(m, Matrix4f(m).transpose());
        }
    };

    template <typename Point>
    struct BoundingBox {
        using Float = value_t<Point>;
        static constexpr auto N = array_size_v<Point>;
        using Vector = akari::Vector<Float, N>;
        Point pmin, pmax;
        AKR_XPU BoundingBox() { reset(); }
        AKR_XPU BoundingBox(const Point &pmin, const Point &pmax) : pmin(pmin), pmax(pmax) {}
        AKR_XPU void reset() {
            pmin = Point(Constants<Float>::Inf());
            pmax = Point(-Constants<Float>::Inf());
        }
        AKR_XPU Vector extents() const {
           return pmax - pmin;
        }
        AKR_XPU bool contains(const Point &p) const { return all(p >= pmin && p <= pmax); }
        AKR_XPU Vector size() const { return extents(); }
        AKR_XPU Vector offset(const Point &p) { return (p - pmin) / extents(); }
        AKR_XPU BoundingBox expand(const Point &p) const { return BoundingBox(min(pmin, p), max(pmax, p)); }
        AKR_XPU BoundingBox merge(const BoundingBox &b1) const { return merge(*this, b1); }
        AKR_XPU static BoundingBox merge(const BoundingBox &b1, const BoundingBox &b2) {
            return BoundingBox(min(b1.pmin, b2.pmin), max(b1.pmax, b2.pmax));
        }
        AKR_XPU BoundingBox intersect(const BoundingBox &rhs) const {
            return BoundingBox(max(pmin, rhs.pmin), min(pmax, rhs.pmax));
        }
        AKR_XPU bool empty() const { return any(pmin > pmax) || hsum( extents()) == 0; }
        AKR_XPU Point centroid() const { return extents() * 0.5f + pmin; }
        AKR_XPU Float surface_area_ratio(const BoundingBox & rhs)const{
            // static_assert(N == 3);
            // auto e1 = extents();
            // auto e2 = rhs.extents();
            // auto s1 = akari::shuffle<1, 2, 0>(e1) * e1;
            // auto s2 = akari::shuffle<1, 2, 0>(e2) * e2;
            // Float r = 1.0f;
            // for(int i =0;i<3;i++){
            //     if(s1[i] <= 0.0 && s2[i] <= 0.0){
            //         auto remap = []AKR_XPU(Float x) ->Float{return x <= 0.0 ? 1.0 : x;};
            //         r *= remap(s1[i]) / remap(s2[i]);
            //     }else{
            //         r *= s1[i] / s2[i];
            //     }
            // }
            // return r < 0.0 ? 0.0 : r;
            return rhs.surface_area() >= 0.0 ? surface_area() / rhs.surface_area() : 0.0;
        }
        AKR_XPU Float surface_area() const {
            if (empty())
                return Float(0.0);
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

    template <typename Float>
    AKR_XPU inline Float degrees(Float x) {
        return x * Constants<value_t<Float>>::InvPi() * 180.0f;
    }
    template <typename Float>
    AKR_XPU inline Float radians(Float x) {
        return x * Constants<value_t<Float>>::Pi() / 180.0f;
    }
} // namespace akari