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
#include <algorithm>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <akari/core/fwd.h>
#include <akari/core/panic.h>
namespace akari {

#define USE_GLM_TVEC(prefix, i) using glm::prefix##vec##i;
#define USE_GLM_VEC_PREFIX(prefix)                                                                                     \
    USE_GLM_TVEC(prefix, 1) USE_GLM_TVEC(prefix, 2) USE_GLM_TVEC(prefix, 3) USE_GLM_TVEC(prefix, 4)
    USE_GLM_VEC_PREFIX(i)
    USE_GLM_VEC_PREFIX(u)
    USE_GLM_VEC_PREFIX(b)
    USE_GLM_VEC_PREFIX(d)
#define USE_GLM_TMAT(prefix, i)    using glm::prefix##mat##i;
#define USE_GLM_MAT_PREFIX(prefix) USE_GLM_TMAT(prefix, 2) USE_GLM_TMAT(prefix, 3) USE_GLM_TMAT(prefix, 4)
    USE_GLM_MAT_PREFIX(d)

    using glm::vec1;
    using glm::vec2;
    using glm::vec3;
    using glm::vec4;

    using glm::mat2;
    using glm::mat3;
    using glm::mat4;

    template <typename T, int N>
    using Vector = glm::vec<N, T, glm::defaultp>;
    template <typename T, int N>
    using Mat = glm::mat<N, N, T, glm::defaultp>;
    using Vec1 = Vector<Float, 1>;
    using Vec2 = Vector<Float, 2>;
    using Vec3 = Vector<Float, 3>;
    using Vec4 = Vector<Float, 4>;

    using Mat2 = Mat<Float, 2>;
    using Mat3 = Mat<Float, 3>;
    using Mat4 = Mat<Float, 4>;

    static constexpr Float Inf = std::numeric_limits<Float>::infinity();
    static constexpr Float Pi = Float(3.1415926535897932384f);
    static constexpr Float Pi2 = Pi / Float(2.0f);
    static constexpr Float Pi4 = Pi / Float(4.0f);
    static constexpr Float InvPi = Float(1.0f) / Pi;
    static constexpr Float Eps = Float(0.001f);
    static constexpr Float ShadowEps = Float(0.0001f);

    template <typename T, typename Float>
    T lerp(T a, T b, Float t) {
        return a * (Float(1.0) - t) + b * t;
    }
    template <typename V, typename V2>
    inline V lerp3(const V &v0, const V &v1, const V &v2, const V2 &uv) {
        return (1.0f - uv[0] - uv[1]) * v0 + uv[0] * v1 + uv[1] * v2;
    }
    template <typename T, int N, class F>
    T reduce(const Vector<T, N> &vec, F &&f) {
        T acc = vec[0];
        for (int i = 1; i < N; i++) {
            acc = f(acc, vec[i]);
        }
        return acc;
    }
    template <int... args, typename T, int N>
    auto shuffle(const Vector<T, N> &a) {
        constexpr int pack[] = {args...};
        static_assert(((args < N) && ...));
        Vector<T, sizeof...(args)> s;
        for (size_t i = 0; i < sizeof...(args); i++) {
            s[i] = a[pack[i]];
        }
        return s;
    }
    template <typename T, int N>
    T hsum(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return acc + cur; });
    }
    template <typename T, int N>
    T hprod(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return acc * cur; });
    }
    using std::min;
    template <typename T, int N>
    T hmin(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return min(acc, cur); });
    }
    using std::max;
    template <typename T, int N>
    T hmax(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return max(acc, cur); });
    }
    template <typename T, int N, class F>
    auto foldl(const Vector<T, N> &vec, T init, F &&f) {
        auto acc = f(init, vec[0]);
        for (int i = 1; i < N; i++) {
            acc = f(acc, vec[i]);
        }
        return acc;
    }
    template <typename T, int N>
    Vector<T, N> select(const Vector<bool, N> &c, const Vector<T, N> &a, const Vector<T, N> &b) {
        return glm::mix(b, a, c);
    }

    template <typename T>
    struct vec_trait {};

    template <typename T, int N>
    struct vec_trait<Vector<T, N>> {
        using value_type = T;
        static constexpr int size = N;
    };

    template <typename T, typename V = typename vec_trait<T>::value_type, int N = vec_trait<T>::size>
    T load(const V *arr) {
        T v;
        for (int i = 0; i < N; i++)
            v[i] = arr[i];
        return v;
    }
    struct Ray {
        // Float time = 0.0f;
        vec3 o;
        vec3 d;
        Float tmin = -1, tmax = -1;
        Ray() = default;
        Ray(const vec3 &o, const vec3 &d, Float tmin = Eps, Float tmax = std::numeric_limits<Float>::infinity())
            : o(o), d(d), tmin(tmin), tmax(tmax) {}
        static Ray spawn_to(const vec3 &p0, const vec3 &p1) {
            vec3 dir = p1 - p0;
            return Ray(p0, dir, Eps, Float(1.0f) - ShadowEps);
        }
        vec3 operator()(Float t) const { return o + t * d; }
    };

    struct Frame {
        Frame() = default;
        static inline void compute_local_frame(const vec3 &v1, vec3 *v2, vec3 *v3) {
            if (std::abs(v1.x) > std::abs(v1.y))
                *v2 = vec3(-v1.z, (0), v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
            else
                *v2 = vec3((0), v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
            *v3 = normalize(cross(vec3(v1), *v2));
        }
        explicit Frame(const vec3 &v) : normal(v) { compute_local_frame(v, &T, &B); }

        [[nodiscard]] vec3 world_to_local(const vec3 &v) const { return vec3(dot(T, v), dot(normal, v), dot(B, v)); }

        [[nodiscard]] vec3 local_to_world(const vec3 &v) const { return vec3(v.x * T + v.y * vec3(normal) + v.z * B); }

        vec3 normal;
        vec3 T, B;
    };

    struct Transform {
        Mat4 m, minv;
        Mat3 m3, m3inv;
        Transform() : Transform(glm::mat4(1.0)) {}
        Transform(const Mat4 &m) : Transform(m, glm::inverse(m)) {}
        Transform(const Mat4 &m, const Mat4 &minv) : m(m), minv(minv) {
            m3 = glm::mat3(m);
            m3inv = glm::inverse(m3);
        }
        Transform inverse() const { return Transform(minv, m); }
        vec3 apply_point(const vec3 &p) const {
            Vec4 v(p.x, p.y, p.z, 1.0);
            v = m * v;
            vec3 q(v.x, v.y, v.z);
            if (v.w != 1.0) {
                q /= v.w;
            }
            return q;
        }
        Transform operator*(const Transform &t) const { return Transform(m * t.m); }
        vec3 apply_vector(const vec3 &v) const { return m3 * v; }
        vec3 apply_normal(const vec3 &n) const { return transpose(m3inv) * n; }

        static Transform translate(const vec3 &v) {
            mat4 m = glm::translate(glm::mat4(1.0), v);
            return Transform(m);
        }
        static Transform scale(const vec3 &s) {
            mat4 m = glm::scale(glm::mat4(1.0), s);
            return Transform(m);
        }
        static Transform rotate_x(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(1, 0, 0));
            return Transform(m, glm::transpose(m));
        }

        static Transform rotate_y(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(0, 1, 0));
            return Transform(m, glm::transpose(m));
        }

        static Transform rotate_z(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(0, 0, 1));
            return Transform(m, glm::transpose(m));
        }
    };

    template <typename T, int N>
    struct BoundingBox {
        using V = Vector<T, N>;
        V pmin, pmax;
        BoundingBox() { reset(); }
        BoundingBox(const V &pmin, const V &pmax) : pmin(pmin), pmax(pmax) {}
        void reset() {
            pmin =V(std::numeric_limits<T>::infinity());
            pmax =V(-std::numeric_limits<T>::infinity());
        }
        V extents() const { return pmax - pmin; }
        bool contains(const V &p) const { return all(p >= pmin && p <= pmax); }
        V size() const { return extents(); }
        V offset(const V &p) { return (p - pmin) / extents(); }
        BoundingBox expand(const V &p) const { return BoundingBox(min(pmin, p), max(pmax, p)); }
        BoundingBox merge(const BoundingBox &b1) const { return merge(*this, b1); }
        static BoundingBox merge(const BoundingBox &b1, const BoundingBox &b2) {
            return BoundingBox(min(b1.pmin, b2.pmin), max(b1.pmax, b2.pmax));
        }
        BoundingBox intersect(const BoundingBox &rhs) const {
            return BoundingBox(max(pmin, rhs.pmin), min(pmax, rhs.pmax));
        }
        bool empty() const { return any(glm::greaterThan(pmin, pmax)) || hsum(extents()) == 0; }
        V centroid() const { return extents() * 0.5f + pmin; }
        Float surface_area() const {
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

    using Bounds2f = BoundingBox<Float, 2>;
    using Bounds3f = BoundingBox<Float, 3>;

    using Bounds2i = BoundingBox<int, 2>;
    using Bounds3i = BoundingBox<int, 3>;
} // namespace akari