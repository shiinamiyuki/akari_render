// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <akari/common.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#ifndef __CUDACC__
#    include <akari/macro.h>
#else
#    define AKR_SER(...)
#endif

namespace akari {
    template <class T>
    AKR_XPU void swap(T &a, T &b) {
        T tmp = a;
        a     = b;
        b     = tmp;
    }
    template <typename T, int N>
    struct Color;
    using Float    = float;
    using Spectrum = Color<Float, 3>;
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
    using Mat  = glm::mat<N, N, T, glm::defaultp>;
    using Vec1 = Vector<Float, 1>;
    using Vec2 = Vector<Float, 2>;
    using Vec3 = Vector<Float, 3>;
    using Vec4 = Vector<Float, 4>;

    using Mat2 = Mat<Float, 2>;
    using Mat3 = Mat<Float, 3>;
    using Mat4 = Mat<Float, 4>;
    // #ifdef AKR_BACKEND_CUDA
    // #    define Inf       (std::numeric_limits<Float>::infinity())
    // #    define Pi        (Float(3.1415926535897932384f))
    // #    define MaxFloat  (std::numeric_limits<Float>::max())
    // #    define PiOver2   (Pi / Float(2.0f))
    // #    define PiOver4   (Pi / Float(4.0f))
    // #    define InvPi     (Float(1.0f) / Pi)
    // #    define Inv2Pi    (Float(1.0) / (2.0 * Pi))
    // #    define Inv4Pi    (Float(1.0) / (4.0 * Pi))
    // #    define Eps       (Float(1e-5f))
    // #    define ShadowEps (Float(0.0001f))
    // #else
    static constexpr Float Inf       = std::numeric_limits<Float>::infinity();
    static constexpr Float MaxFloat  = std::numeric_limits<Float>::max();
    static constexpr Float Pi        = Float(3.1415926535897932384f);
    static constexpr Float PiOver2   = Pi / Float(2.0f);
    static constexpr Float PiOver4   = Pi / Float(4.0f);
    static constexpr Float InvPi     = Float(1.0f) / Pi;
    static constexpr Float Inv2Pi    = Float(1.0) / (2.0 * Pi);
    static constexpr Float Inv4Pi    = Float(1.0) / (4.0 * Pi);
    static constexpr Float Eps       = Float(1e-5f);
    static constexpr Float ShadowEps = Float(0.0001f);

    static constexpr Float MachineEpsilon = std::numeric_limits<Float>::epsilon() * 0.5;

    static constexpr double DoubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
    static constexpr float FloatOneMinusEpsilon   = 0x1.fffffep-1;

    static constexpr float OneMinusEpsilon = FloatOneMinusEpsilon;
    // #endif
    template <typename T, typename Float>
    AKR_XPU T lerp(T a, T b, Float t) {
        return a * ((Float(1.0)) - t) + b * t;
    }
    template <typename V, typename V2>
    AKR_XPU inline V lerp3(const V &v0, const V &v1, const V &v2, const V2 &uv) {
        return (1.0f - uv[0] - uv[1]) * v0 + uv[0] * v1 + uv[1] * v2;
    }
    template <typename V, typename V2>
    AKR_XPU inline V dlerp3du(const V &v0, const V &v1, const V &v2, V2 u) {
        return -v0 + v1;
    }
    template <typename V, typename V2>
    AKR_XPU inline V dlerp3dv(const V &v0, const V &v1, const V &v2, V2 v) {
        return -v0 + v2;
    }
    template <typename T, int N, class F>
    AKR_XPU T map(const Vector<T, N> &vec, F &&f) {
        Vector<T, N> out;
        for (int i = 0; i < N; i++) {
            out[i] = f(vec[i]);
        }
        return out;
    }
    template <typename T, int N, class F>
    AKR_XPU T reduce(const Vector<T, N> &vec, F &&f) {
        T acc = vec[0];
        for (int i = 1; i < N; i++) {
            acc = f(acc, vec[i]);
        }
        return acc;
    }
    template <int... args, typename T, int N>
    AKR_XPU auto shuffle(const Vector<T, N> &a) {
        constexpr int pack[] = {args...};
        static_assert(((args < N) && ...));
        Vector<T, sizeof...(args)> s;
        for (size_t i = 0; i < sizeof...(args); i++) {
            s[i] = a[pack[i]];
        }
        return s;
    }
    template <typename T, int N>
    AKR_XPU T hsum(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return acc + cur; });
    }
    template <typename T, int N>
    AKR_XPU T hprod(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return acc * cur; });
    }
    using std::min;
    template <typename T, int N>
    AKR_XPU T hmin(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return min(acc, cur); });
    }
    using std::max;
    template <typename T, int N>
    AKR_XPU T hmax(const Vector<T, N> &vec) {
        return reduce(vec, [](T acc, T cur) -> T { return max(acc, cur); });
    }
    template <typename T, int N, typename R, class F>
    AKR_XPU R foldl(const Vector<T, N> &vec, R init, F &&f) {
        auto acc = f(init, vec[0]);
        for (int i = 1; i < N; i++) {
            acc = f(acc, vec[i]);
        }
        return acc;
    }
    template <typename T, int N>
    AKR_XPU Vector<T, N> select(const Vector<bool, N> &c, const Vector<T, N> &a, const Vector<T, N> &b) {
        return glm::mix(b, a, c);
    }
    template <typename T, int N>
    AKR_XPU inline auto l2norm(const Vector<T, N> &a, const Vector<T, N> &b) {
        auto v = a - b;
        return dot(v, v);
    }
    template <typename T>
    struct vec_trait {
        using value_type                = void;
        static constexpr bool is_vector = false;
    };

    template <typename T, int N>
    struct vec_trait<Vector<T, N>> {
        using value_type                = T;
        static constexpr int size       = N;
        static constexpr bool is_vector = true;
    };

    template <typename T, typename V = typename vec_trait<T>::value_type, int N = vec_trait<T>::size>
    AKR_XPU T load(const V *arr) {
        T v;
        for (int i = 0; i < N; i++)
            v[i] = arr[i];
        return v;
    }

    template <typename Scalar, int N>
    struct Color : Vector<Scalar, N> {
        using Base = Vector<Scalar, N>;
        using Base::Base;
        using value_t                = Scalar;
        static constexpr size_t size = N;
        AKR_XPU Color(const Base &v) : Base(v) {}
#define AKR_COLOR_OP(op)                                                                                               \
    AKR_XPU Color operator op(const Color &rhs) const { return Color(Base(*this) op Base(rhs)); }                      \
    AKR_XPU Color operator op(Scalar rhs) const { return Color(Base(*this) op Base(rhs)); }                            \
    AKR_XPU friend Color operator op(Scalar lhs, const Color &rhs) { return Color(Base(lhs) op Base(rhs)); }           \
    AKR_XPU Color &operator op##=(const Color &rhs) {                                                                  \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }                                                                                                                  \
    AKR_XPU Color &operator op##=(Scalar rhs) {                                                                        \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }
        AKR_COLOR_OP(+) AKR_COLOR_OP(-) AKR_COLOR_OP(*) AKR_COLOR_OP(/)
#undef AKR_COLOR_OP
    };
    template <typename Scalar, int N>
    AKR_XPU Color<Scalar, N> clamp_zero(const Color<Scalar, N> &in) {
        Color<Scalar, N> c;
        for (int i = 0; i < N; i++) {
            auto x = in[i];
            if (std::isnan(x)) {
                x = 0;
            } else {
                x = max(Scalar(0.0f), x);
            }
            c[i] = x;
        }
        return c;
    }
    template <typename Scalar, int N>
    AKR_XPU Color<Scalar, N> exp(const Color<Scalar, N> &in) {
        return map(in, [](Scalar x) -> Scalar { return exp(x); });
    }
    template <typename Scalar, int N>
    AKR_XPU Color<Scalar, N> min(const Color<Scalar, N> &in, const Color<Scalar, N> &v) {
        Color<Scalar, N> c;
        for (int i = 0; i < N; i++) {
            c[i] = std::min(in[i], v[i]);
        }
        return c;
    }
    template <typename Scalar, int N>
    AKR_XPU bool is_black(const Color<Scalar, N> &color) {
        return !foldl(color, false, [](bool acc, Scalar f) { return acc || (f > 0.0f); });
    }
    inline Float linear_to_srgb(Float L) {
        return (L < 0.0031308) ? (L * 12.92) : (1.055 * std::pow(L, 1.0 / 2.4) - 0.055);
    }
    inline Float srgb_to_linear(Float S) { return (S < 0.04045) ? (S / 12.92) : (std::pow(S + 0.055, 2.4) / 1.055); }
    template <typename Scalar>
    AKR_XPU Color<Scalar, 3> linear_to_srgb(const Color<Scalar, 3> &L) {
        using Color3f = Color<Scalar, 3>;
        return select(glm::lessThan(L, Color3f(0.0031308)), L * 12.92,
                      Float(1.055) * glm::pow(L, Vec3(1.0f / 2.4f)) - Float(0.055));
    }
    template <typename Scalar>
    AKR_XPU Color<Scalar, 3> srgb_to_linear(const Color<Scalar, 3> &S) {
        using Color3f = Color<Scalar, 3>;
        return select(glm::lessThan(S, Color3f(0.04045)), S / 12.92, glm::pow((S + 0.055) / 1.055, Vec3(2.4)));
    }

    using Color3f = Color<Float, 3>;

    AKR_XPU inline Float luminance(const Color3f &rgb) { return dot(rgb, Vec3(0.2126, 0.7152, 0.0722)); }
    AKR_XPU inline Float average(const Color3f &rgb) { return hsum(rgb) / 3.0f; }
    template <typename T, int N>
    struct vec_trait<Color<T, N>> {
        using value_type                = T;
        static constexpr int size       = N;
        static constexpr bool is_vector = true;
    };

    /*
    A Row major 2x2 matrix
    */
    template <typename T>
    struct Matrix2 {
        AKR_XPU Matrix2(Mat<T, 2> m) : m(m) {}
        AKR_XPU Matrix2() { m = glm::identity<Mat<T, 2>>(); }
        AKR_XPU Matrix2(T x00, T x01, T x10, T x11) { m = Mat<T, 2>(x00, x10, x01, x11); }
        Matrix2 inverse() const { return Matrix2(glm::inverse(m)); }
        AKR_XPU Vector<T, 2> operator*(const Vector<T, 2> &v) const { return m * v; }
        AKR_XPU Matrix2<T> operator*(const Matrix2<T> &n) const { return Matrix2(m * n.m); }
        AKR_XPU T &operator()(int i, int j) { return m[j][i]; }
        AKR_XPU const T &operator()(int i, int j) const { return m[j][i]; }
        AKR_XPU Float determinant() const { return glm::determinant(m); }

      private:
        Mat<T, 2> m;
    };
    using Matrix2f = Matrix2<float>;
    using Matrix2d = Matrix2<double>;

    struct Ray {
        // Float time = 0.0f;
        vec3 o;
        vec3 d;
        Float tmin         = -1;
        mutable Float tmax = -1;
        Ray()              = default;
        AKR_XPU Ray(const vec3 &o, const vec3 &d, Float tmin = Eps, Float tmax = std::numeric_limits<Float>::infinity())
            : o(o), d(d), tmin(tmin), tmax(tmax) {}
        AKR_XPU vec3 operator()(Float t) const { return o + t * d; }
        AKR_SER(o, d, tmin, tmax)
    };

    namespace robust_rt {
        AKR_XPU constexpr inline float origin() { return 1.0f / 32.0f; }
        AKR_XPU constexpr inline float float_scale() { return 1.0f / 65536.0f; }
        AKR_XPU constexpr inline float int_scale() { return 256.0f; }

    } // namespace robust_rt
    // Normal points outward for rays exiting the surface, else is flipped.
    AKR_XPU inline vec3 offset_ray(const vec3 p, const vec3 n) {
        using namespace robust_rt;
        ivec3 of_i(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

        vec3 p_i(glm::intBitsToFloat(glm::floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                 glm::intBitsToFloat(glm::floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                 glm::intBitsToFloat(glm::floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

        return vec3(fabsf(p.x) < origin() ? p.x + float_scale() * n.x : p_i.x,
                    fabsf(p.y) < origin() ? p.y + float_scale() * n.y : p_i.y,
                    fabsf(p.z) < origin() ? p.z + float_scale() * n.z : p_i.z);
    }

    AKR_XPU inline glm::dvec3 offset_ray(const glm::dvec3 p, const glm::dvec3 n) { return p; }
    AKR_XPU inline Ray spawn_ray(const Vec3 &o, const Vec3 &d, const Vec3 &n) {
        auto ray = Ray(o, d, Eps);
        ray.o    = offset_ray(ray.o, dot(d, n) > 0 ? n : -n);
        return ray;
    }
    AKR_XPU inline Ray spawn_to(const Vec3 &p1, const Vec3 &p2, const Vec3 &n) {
        auto w    = p2 - p1;
        auto dist = length(w);
        w /= dist;
        auto ray = spawn_ray(p1, w, n);
        ray.tmax = dist * (1.0 - 1e-7f);
        return ray;
    }
    struct Frame {
        Frame() = default;
        AKR_XPU static inline void compute_local_frame(const vec3 &v1, vec3 *v2, vec3 *v3) {
            if (std::abs(v1.x) > std::abs(v1.y))
                *v2 = vec3(-v1.z, (0), v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
            else
                *v2 = vec3((0), v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
            *v3 = normalize(cross(vec3(v1), *v2));
        }
        AKR_XPU explicit Frame(const vec3 &n) : n(n) { compute_local_frame(n, &s, &t); }
        AKR_XPU explicit Frame(const vec3 &n, const vec3 &dpdu) : n(n) {
            s = glm::normalize(-n * glm::dot(n, dpdu) + dpdu);
            t = glm::cross(n, s);
        }
        [[nodiscard]] AKR_XPU vec3 world_to_local(const vec3 &v) const { return vec3(dot(s, v), dot(n, v), dot(t, v)); }

        [[nodiscard]] AKR_XPU vec3 local_to_world(const vec3 &v) const {
            return vec3(v.x * s + v.y * vec3(n) + v.z * t);
        }

        vec3 n;
        vec3 s, t;
    };

    struct Transform {
        Mat4 m, minv;
        Mat3 m3, m3inv;
        AKR_SER(m, minv, m3, m3inv)
        AKR_XPU Transform() : Transform(glm::mat4(1.0)) {}
        AKR_XPU Transform(const Mat4 &m) : Transform(m, glm::inverse(m)) {}
        AKR_XPU Transform(const Mat4 &m, const Mat4 &minv) : m(m), minv(minv) {
            m3    = glm::mat3(m);
            m3inv = glm::inverse(m3);
        }
        AKR_XPU Transform inverse() const { return Transform(minv, m); }
        AKR_XPU vec3 apply_point(const vec3 &p) const {
            Vec4 v(p.x, p.y, p.z, 1.0);
            v = m * v;
            vec3 q(v.x, v.y, v.z);
            if (v.w != 1.0) {
                q /= v.w;
            }
            return q;
        }
        AKR_XPU Transform operator*(const Transform &t) const { return Transform(m * t.m); }
        AKR_XPU vec3 apply_vector(const vec3 &v) const { return m3 * v; }
        AKR_XPU vec3 apply_normal(const vec3 &n) const { return transpose(m3inv) * n; }

        AKR_XPU static Transform translate(const vec3 &v) {
            mat4 m = glm::translate(glm::mat4(1.0), v);
            return Transform(m);
        }
        AKR_XPU static Transform scale(const vec3 &s) {
            mat4 m = glm::scale(glm::mat4(1.0), s);
            return Transform(m);
        }
        AKR_XPU static Transform rotate_x(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(1, 0, 0));
            return Transform(m, glm::transpose(m));
        }

        AKR_XPU static Transform rotate_y(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(0, 1, 0));
            return Transform(m, glm::transpose(m));
        }

        AKR_XPU static Transform rotate_z(Float theta) {
            auto m = glm::rotate(glm::mat4(1.0), theta, vec3(0, 0, 1));
            return Transform(m, glm::transpose(m));
        }
    };

    template <typename T, int N>
    struct BoundingBox {
        using V = Vector<T, N>;
        V pmin, pmax;
        AKR_XPU BoundingBox() { reset(); }
        AKR_XPU BoundingBox(const V &pmin, const V &pmax) : pmin(pmin), pmax(pmax) {}
        AKR_XPU void reset() {
            pmin = V(std::numeric_limits<T>::infinity());
            pmax = V(-std::numeric_limits<T>::infinity());
        }
        AKR_XPU V extents() const { return pmax - pmin; }
        AKR_XPU bool contains(const V &p) const { return all(p >= pmin && p <= pmax); }
        AKR_XPU V size() const { return extents(); }
        AKR_XPU V offset(const V &p) { return (p - pmin) / extents(); }
        AKR_XPU BoundingBox expand(const V &p) const { return BoundingBox(min(pmin, p), max(pmax, p)); }
        AKR_XPU BoundingBox merge(const BoundingBox &b1) const { return merge(*this, b1); }
        AKR_XPU static BoundingBox merge(const BoundingBox &b1, const BoundingBox &b2) {
            return BoundingBox(min(b1.pmin, b2.pmin), max(b1.pmax, b2.pmax));
        }
        AKR_XPU BoundingBox intersect(const BoundingBox &rhs) const {
            return BoundingBox(max(pmin, rhs.pmin), min(pmax, rhs.pmax));
        }
        AKR_XPU V clip(const V &p) const { return min(max(p, pmin), pmax); }
        AKR_XPU bool empty() const { return any(glm::greaterThan(pmin, pmax)) || hsum(extents()) == 0; }
        AKR_XPU V centroid() const { return extents() * 0.5f + pmin; }
        AKR_XPU Float surface_area() const {
            if (empty())
                return Float(0.0);
            if constexpr (N == 3) {
                auto ext = extents();
                return hsum(akari::shuffle<1, 2, 0>(ext) * ext) * Float(2);
            } else {
                auto ext     = extents();
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

namespace akari::render {
#pragma region sampling
    AKR_XPU inline glm::vec2 concentric_disk_sampling(const glm::vec2 &u) {
        glm::vec2 uOffset = ((float(2.0) * u) - glm::vec2(int32_t(1), int32_t(1)));
        if (((uOffset.x == float(0.0)) && (uOffset.y == float(0.0))))
            return glm::vec2(int32_t(0), int32_t(0));
        float theta = float();
        float r     = float();
        if ((glm::abs(uOffset.x) > glm::abs(uOffset.y))) {
            r     = uOffset.x;
            theta = (PiOver4 * (uOffset.y / uOffset.x));
        } else {
            r     = uOffset.y;
            theta = (PiOver2 - (PiOver4 * (uOffset.x / uOffset.y)));
        }
        return (r * glm::vec2(glm::cos(theta), glm::sin(theta)));
    }
    AKR_XPU inline glm::vec3 cosine_hemisphere_sampling(const glm::vec2 &u) {
        glm::vec2 uv = concentric_disk_sampling(u);
        float r      = glm::dot(uv, uv);
        float h      = glm::sqrt(glm::max(float(float(0.0)), float((float(1.0) - r))));
        return glm::vec3(uv.x, h, uv.y);
    }
    AKR_XPU inline float cosine_hemisphere_pdf(float cosTheta) { return (cosTheta * InvPi); }
    AKR_XPU inline float uniform_sphere_pdf() { return (float(1.0) / (float(4.0) * Pi)); }
    AKR_XPU inline glm::vec3 uniform_sphere_sampling(const glm::vec2 &u) {
        float z   = (float(1.0) - (float(2.0) * u[int32_t(0)]));
        float r   = glm::sqrt(glm::max(float(0.0), (float(1.0) - (z * z))));
        float phi = ((float(2.0) * Pi) * u[int32_t(1)]);
        return glm::vec3((r * glm::cos(phi)), (r * glm::sin(phi)), z);
    }
    AKR_XPU inline glm::vec2 uniform_sample_triangle(const glm::vec2 &u) {
        // float su0 = glm::sqrt(u[int32_t(0)]);
        // float b0 = (float(1.0) - su0);
        // float b1 = (u[int32_t(1)] * su0);
        // return glm::vec2(b0, b1);

        uint32_t uf = u[0] * (1ull << 32); // Fixed point
        Float cx = 0.0f, cy = 0.0f;
        Float w = 0.5f;

        for (int i = 0; i < 16; i++) {
            uint32_t uu = uf >> 30;
            bool flip   = (uu & 3) == 0;

            cy += ((uu & 1) == 0) * w;
            cx += ((uu & 2) == 0) * w;

            w *= flip ? -0.5f : 0.5f;
            uf <<= 2;
        }

        Float b0 = cx + w / 3.0f, b1 = cy + w / 3.0f;
        return vec2(b0, b1);
    }
#pragma endregion
#pragma region geometry
    AKR_XPU inline float cos_theta(const glm::vec3 &w) { return w.y; }
    AKR_XPU inline float abs_cos_theta(const glm::vec3 &w) { return glm::abs(cos_theta(w)); }
    AKR_XPU inline float cos2_theta(const glm::vec3 &w) { return (w.y * w.y); }
    AKR_XPU inline float sin2_theta(const glm::vec3 &w) { return (float(1.0) - cos2_theta(w)); }
    AKR_XPU inline float sin_theta(const glm::vec3 &w) { return glm::sqrt(glm::max(float(0.0), sin2_theta(w))); }
    AKR_XPU inline float tan2_theta(const glm::vec3 &w) { return (sin2_theta(w) / cos2_theta(w)); }
    AKR_XPU inline float tan_theta(const glm::vec3 &w) { return glm::sqrt(glm::max(float(0.0), tan2_theta(w))); }
    AKR_XPU inline float cos_phi(const glm::vec3 &w) {
        float sinTheta = sin_theta(w);
        return (sinTheta == float(0.0)) ? float(1.0) : glm::clamp((w.x / sinTheta), -float(1.0), float(1.0));
    }
    AKR_XPU inline float sin_phi(const glm::vec3 &w) {
        float sinTheta = sin_theta(w);
        return (sinTheta == float(0.0)) ? float(0.0) : glm::clamp((w.z / sinTheta), -float(1.0), float(1.0));
    }
    AKR_XPU inline float cos2_phi(const glm::vec3 &w) { return (cos_phi(w) * cos_phi(w)); }
    AKR_XPU inline float sin2_phi(const glm::vec3 &w) { return (sin_phi(w) * sin_phi(w)); }
    AKR_XPU inline bool same_hemisphere(const glm::vec3 &wo, const glm::vec3 &wi) {
        return ((wo.y * wi.y) >= float(0.0));
    }
    AKR_XPU inline vec3 refract(const glm::vec3 &wi, const glm::vec3 &n, float eta) {
        float cosThetaI  = glm::dot(n, wi);
        float sin2ThetaI = glm::max(float(0.0), (float(1.0) - (cosThetaI * cosThetaI)));
        float sin2ThetaT = ((eta * eta) * sin2ThetaI);
        if ((sin2ThetaT >= float(1.0)))
            return vec3(0);
        float cosThetaT = glm::sqrt((float(1.0) - sin2ThetaT));
        auto wt         = ((eta * -wi) + (((eta * cosThetaI) - cosThetaT) * n));
        return wt;
    }
    AKR_XPU inline vec3 faceforward(const vec3 &w, const vec3 &n) { return dot(w, n) < 0.0 ? -n : n; }
    AKR_XPU inline float fr_dielectric(float cosThetaI, float etaI, float etaT) {
        bool entering = (cosThetaI > float(0.0));
        if (!entering) {
            swap(etaI, etaT);
            cosThetaI = glm::abs(cosThetaI);
        }
        float sinThetaI = glm::sqrt(glm::max(float(0.0), (float(1.0) - (cosThetaI * cosThetaI))));
        float sinThetaT = ((etaI / etaT) * sinThetaI);
        if ((sinThetaT >= float(1.0)))
            return float(1.0);
        float cosThetaT = glm::sqrt(glm::max(float(0.0), (float(1.0) - (sinThetaT * sinThetaT))));
        float Rpar      = (((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT)));
        float Rper      = (((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT)));
        return (float(0.5) * ((Rpar * Rpar) + (Rper * Rper)));
    }
    AKR_XPU inline glm::vec3 fr_conductor(float cosThetaI, const glm::vec3 &etaI, const glm::vec3 &etaT,
                                          const glm::vec3 &k) {
        float CosTheta2    = (cosThetaI * cosThetaI);
        float SinTheta2    = (float(1.0) - CosTheta2);
        glm::vec3 Eta      = (etaT / etaI);
        glm::vec3 Etak     = (k / etaI);
        glm::vec3 Eta2     = (Eta * Eta);
        glm::vec3 Etak2    = (Etak * Etak);
        glm::vec3 t0       = ((Eta2 - Etak2) - SinTheta2);
        glm::vec3 a2plusb2 = glm::sqrt(((t0 * t0) + ((float(4.0) * Eta2) * Etak2)));
        glm::vec3 t1       = (a2plusb2 + CosTheta2);
        glm::vec3 a        = glm::sqrt((float(0.5) * (a2plusb2 + t0)));
        glm::vec3 t2       = ((float(2.0) * a) * cosThetaI);
        glm::vec3 Rs       = ((t1 - t2) / (t1 + t2));
        glm::vec3 t3       = ((CosTheta2 * a2plusb2) + (SinTheta2 * SinTheta2));
        glm::vec3 t4       = (t2 * SinTheta2);
        glm::vec3 Rp       = ((Rs * (t3 - t4)) / (t3 + t4));
        return (float(0.5) * (Rp + Rs));
    }

    AKR_XPU inline vec3 spherical_to_xyz(float sinTheta, float cosTheta, float phi) {
        return glm::vec3(sinTheta * glm::cos(phi), cosTheta, sinTheta * glm::sin(phi));
    }

    AKR_XPU inline float spherical_theta(const vec3 &v) { return glm::acos(glm::clamp(v.y, -1.0f, 1.0f)); }

    AKR_XPU inline float spherical_phi(const glm::vec3 v) {
        float p = glm::atan(v.z, v.x);
        return p < 0.0 ? (p + 2.0 * Pi) : p;
    }
#pragma endregion
#pragma region

    static const int32_t MicrofacetGGX      = int32_t(0);
    static const int32_t MicrofacetBeckmann = int32_t(1);
    static const int32_t MicrofacetPhong    = int32_t(2);
    AKR_XPU inline float BeckmannD(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        float c  = cos2_theta(m);
        float t  = tan2_theta(m);
        float a2 = (alpha * alpha);
        return (glm::exp((-t / a2)) / (((Pi * a2) * c) * c));
    }
    AKR_XPU inline float BeckmannG1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        float a = (float(1.0) / (alpha * tan_theta(v)));
        if ((a < float(1.6))) {
            return (((float(3.535) * a) + ((float(2.181) * a) * a)) /
                    ((float(1.0) + (float(2.276) * a)) + ((float(2.577) * a) * a)));
        } else {
            return float(1.0);
        }
    }
    AKR_XPU inline float PhongG1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        float a = (glm::sqrt(((float(0.5) * alpha) + float(1.0))) / tan_theta(v));
        if ((a < float(1.6))) {
            return (((float(3.535) * a) + ((float(2.181) * a) * a)) /
                    ((float(1.0) + (float(2.276) * a)) + ((float(2.577) * a) * a)));
        } else {
            return float(1.0);
        }
    }
    AKR_XPU inline float PhongD(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        return (((alpha + float(2.0)) / (float(2.0) * Pi)) * glm::pow(m.y, alpha));
    }
    AKR_XPU inline float GGX_D(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        float a2 = (alpha * alpha);
        float c2 = cos2_theta(m);
        float t2 = tan2_theta(m);
        float at = (a2 + t2);
        return (a2 / ((((Pi * c2) * c2) * at) * at));
    }
    AKR_XPU inline float GGX_G1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        return (float(2.0) / (float(1.0) + glm::sqrt((float(1.0) + ((alpha * alpha) * tan2_theta(m))))));
    }
    struct MicrofacetModel {
        int32_t type;
        float alpha;
    };
    AKR_XPU inline MicrofacetModel microfacet_new(int32_t type, float roughness) {
        float alpha = float();
        if ((type == MicrofacetPhong)) {
            alpha = ((float(2.0) / (roughness * roughness)) - float(2.0));
        } else {
            alpha = roughness;
        }
        return MicrofacetModel{type, alpha};
    }
    AKR_XPU inline float microfacet_D(const MicrofacetModel &model, const glm::vec3 &m) {
        int32_t type = model.type;
        float alpha  = model.alpha;
        switch (type) {
        case MicrofacetBeckmann: {
            return BeckmannD(alpha, m);
        }
        case MicrofacetPhong: {
            return PhongD(alpha, m);
        }
        case MicrofacetGGX: {
            return GGX_D(alpha, m);
        }
        }
        return float(0.0);
    }
    AKR_XPU inline float microfacet_G1(const MicrofacetModel &model, const glm::vec3 &v, const glm::vec3 &m) {
        int32_t type = model.type;
        float alpha  = model.alpha;
        switch (type) {
        case MicrofacetBeckmann: {
            return BeckmannG1(alpha, v, m);
        }
        case MicrofacetPhong: {
            return PhongG1(alpha, v, m);
        }
        case MicrofacetGGX: {
            return GGX_G1(alpha, v, m);
        }
        }
        return float(0.0);
    }
    AKR_XPU inline float microfacet_G(const MicrofacetModel &model, const glm::vec3 &i, const glm::vec3 &o,
                                      const glm::vec3 &m) {
        return (microfacet_G1(model, i, m) * microfacet_G1(model, o, m));
    }
    AKR_XPU inline glm::vec3 microfacet_sample_wh(const MicrofacetModel &model, const glm::vec3 &wo,
                                                  const glm::vec2 &u) {
        int32_t type   = model.type;
        float alpha    = model.alpha;
        float phi      = ((float(2.0) * Pi) * u.y);
        float cosTheta = float(0.0);
        switch (type) {
        case MicrofacetBeckmann: {
            float t2 = ((-alpha * alpha) * glm::log((float(1.0) - u.x)));
            cosTheta = (float(1.0) / glm::sqrt((float(1.0) + t2)));
            break;
        }
        case MicrofacetPhong: {
            cosTheta = glm::pow(u.x, float((float(1.0) / (alpha + float(2.0)))));
            break;
        }
        case MicrofacetGGX: {
            float t2 = (((alpha * alpha) * u.x) / (float(1.0) - u.x));
            cosTheta = (float(1.0) / glm::sqrt((float(1.0) + t2)));
            break;
        }
        }
        float sinTheta = glm::sqrt(glm::max(float(0.0), (float(1.0) - (cosTheta * cosTheta))));
        glm::vec3 wh   = glm::vec3((glm::cos(phi) * sinTheta), cosTheta, (glm::sin(phi) * sinTheta));
        if (!same_hemisphere(wo, wh))
            wh = -wh;
        return wh;
    }
    AKR_XPU inline float microfacet_evaluate_pdf(const MicrofacetModel &m, const glm::vec3 &wh) {
        return (microfacet_D(m, wh) * abs_cos_theta(wh));
    }
} // namespace akari::render