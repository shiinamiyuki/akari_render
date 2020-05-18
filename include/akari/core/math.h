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

#ifndef AKARIRENDER_MATH_H
#define AKARIRENDER_MATH_H

#include <json.hpp>
#include <akari/core/config.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <akari/core/simd.hpp>
#include <akari/core/config.h>
#include <akari/core/platform.h>

namespace akari::math {
    using namespace glm;

} // namespace akari::math

namespace glm {
    template <int N, typename T, qualifier Q> void from_json(const nlohmann::json &j, glm::vec<N, T, Q> &vec) {
        for (int i = 0; i < N; i++) {
            vec[i] = j[i].get<T>();
        }
    }

    template <int N, typename T, qualifier Q> void to_json(nlohmann::json &j, const glm::vec<N, T, Q> &vec) {
        for (int i = 0; i < N; i++) {
            j[i] = vec[i];
        }
    }
} // namespace glm

namespace akari {
    using namespace math;
    using namespace nlohmann;
    AKR_EXPORT Float Eps();
    AKR_EXPORT Float ShadowEps();
    constexpr Float Pi = 3.1415926535897932f;
    constexpr Float Pi2 = Pi * 0.5f;
    constexpr Float Pi4 = Pi * 0.25f;
    constexpr Float InvPi = 1.0f / Pi;
    constexpr Float Inv4Pi = 1.0f / (4.0f * Pi);
    constexpr Float Inf = std::numeric_limits<Float>::infinity();
    constexpr Float MaxFloat = std::numeric_limits<Float>::max();
    constexpr Float MinFloat = std::numeric_limits<Float>::min();
    constexpr Float MachineEpsilon = std::numeric_limits<Float>::epsilon();
    template <typename T> inline T RadiansToDegrees(const T &rad) { return rad * (180.0f / Pi); }
    template <typename T> inline T DegreesToRadians(const T &rad) { return rad * (Pi / 180.0f); }
    template <typename T> struct Angle { T value; };

    using EulerAngle = Angle<vec3>;

    template <typename T> void from_json(const json &j, Angle<T> &angle) {
        if (j.contains("deg")) {
            angle.value = DegreesToRadians(j.at("deg").get<T>());
        } else {
            angle.value = j.at("rad").get<T>();
        }
    }

    template <typename T> void to_json(json &j, const Angle<T> &angle) { j["deg"] = RadiansToDegrees(angle.value); }
    struct TransformManipulator {
        EulerAngle rotation;
        vec3 translation;
        vec3 scale = vec3(1);

        [[nodiscard]] mat4 matrix4() const {
            auto m = identity<mat4>();
            m = math::scale(mat4(1.0), scale) * m;
            m = math::rotate(mat4(1.0), rotation.value.z, vec3(0, 0, 1)) * m;
            m = math::rotate(mat4(1.0), rotation.value.y, vec3(1, 0, 0)) * m;
            m = math::rotate(mat4(1.0), rotation.value.x, vec3(0, 1, 0)) * m;
            m = math::translate(mat4(1.0), translation) * m;
            return m;
        }
    };

    inline void from_json(const json &j, TransformManipulator &transform) {
        transform.rotation = j.at("rotation").get<EulerAngle>();
        transform.translation = j.at("translation").get<vec3>();
        transform.scale = j.at("scale").get<vec3>();
    }

    inline void to_json(json &j, const TransformManipulator &transform) {
        j[("rotation")] = transform.rotation;
        j[("translation")] = transform.translation;
        j[("scale")] = transform.scale;
    }

    struct Transform {
        Transform() = default;

        explicit Transform(const mat4 &m)
            : t(m), invt(glm::inverse(m)), t3(mat3(m)), invt3(glm::inverse(mat3(m))), invt3t(transpose(invt3)) {}

        [[nodiscard]] vec3 apply_point(const vec3 &p) const {
            auto q = t * vec4(p, 1);
            if (q.w != 1) {
                return vec3(q) / q.w;
            }
            return vec3(q);
        }

        [[nodiscard]] vec3 apply_normal(const vec3 &n) const { return invt3t * n; }

        [[nodiscard]] vec3 apply_vector(const vec3 &v) const { return t3 * v; }

        [[nodiscard]] Transform inverse() const { return Transform(invt); }

      private:
        mat4 t, invt;
        mat3 t3, invt3, invt3t;
    };

    inline auto min_comp(const vec3 &v) { return min(min(v.x, v.y), v.z); }

    inline auto max_comp(const vec3 &v) { return max(max(v.x, v.y), v.z); }

    template <typename T, int N> struct BoundingBox {
        using P = glm::vec<N, T, glm::defaultp>;
        P p_min, p_max;

        BoundingBox union_of(const BoundingBox &box) const {
            return BoundingBox{min(p_min, box.p_min), max(p_max, box.p_max)};
        }

        BoundingBox union_of(const P &rhs) const { return BoundingBox{min(p_min, rhs), max(p_max, rhs)}; }

        P centroid() const { return size() * 0.5f + p_min; }

        P size() const { return p_max - p_min; }

        T surface_area() const {
            auto a = (size()[0] * size()[1] + size()[0] * size()[2] + size()[1] * size()[2]);
            return a + a;
        }

        bool intersects(const BoundingBox &rhs) const {
            for (size_t i = 0; i < N; i++) {
                if (p_min[i] > rhs.p_max[i] || p_max[i] < rhs.p_min[i])
                    ;
                else {
                    return true;
                }
            }
            return false;
        }

        BoundingBox intersection_of(const BoundingBox &rhs) const {
            return BoundingBox{max(p_min, rhs.p_min), min(p_max, rhs.p_max)};
        }

        P offset(const P &p) const {
            auto o = p - p_min;
            return o / size();
        }
    };

    using Bounds3f = BoundingBox<float, 3>;
    using Bounds2i = BoundingBox<int, 2>;

    inline void compute_local_frame(const vec3 &v1, vec3 *v2, vec3 *v3) {
        if (std::abs(v1.x) > std::abs(v1.y))
            *v2 = vec3(-v1.z, (0), v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
        else
            *v2 = vec3((0), v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
        *v3 = normalize(cross(v1, *v2));
    }

    struct CoordinateSystem {

        CoordinateSystem() = default;

        explicit CoordinateSystem(const vec3 &v) : normal(v) { compute_local_frame(v, &T, &B); }

        [[nodiscard]] vec3 world_to_local(const vec3 &v) const { return vec3(dot(T, v), dot(normal, v), dot(B, v)); }

        [[nodiscard]] vec3 local_to_world(const vec3 &v) const { return vec3(v.x * T + v.y * normal + v.z * B); }

        vec3 normal;
        vec3 T, B;
    };

    template <typename V, typename V2> inline V Interpolate(const V &v0, const V &v1, const V &v2, const V2 &uv) {
        return (1.0f - uv.x - uv.y) * v0 + uv.x * v1 + uv.y * v2;
    }
    template<typename V>
    inline V lerp(const V& a, const V & b, const V& t){
        return (V(1.0f) - t) * a + t * b;
    }
    template <int N, typename T> T Power(const T &x) {
        if constexpr (N == 0) {
            return T(1);
        } else if constexpr (N % 2 == 0) {
            auto tmp = Power<N / 2>(x);
            return tmp * tmp;
        } else {
            auto tmp = Power<N / 2>(x);
            return tmp * tmp * x;
        }
    }

    inline vec3 face_forward(const vec3 &n, const vec3 &w) { return dot(n, w) < 0 ? -n : n; }
} // namespace akari
#endif // AKARIRENDER_MATH_H
