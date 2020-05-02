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

#ifndef AKARIRENDER_GEOMETRY_HPP
#define AKARIRENDER_GEOMETRY_HPP

#include <akari/Core/Component.h>
#include <akari/Core/Math.h>

namespace akari {
    template <typename Value> struct TRay {
        using Vector3 = vec<3, Value>;
        Vector3 o, d;
        Value t_min, t_max;

        TRay() = default;

        TRay(const Vector3 &o, const Vector3 &d, Value t_min = Eps(),
             Value t_max = std::numeric_limits<float>::infinity())
            : o(o), d(d), t_min(t_min), t_max(t_max) {}

        [[nodiscard]] Vector3 At(Value t) const { return o + t * d; }
    };

    using Ray = TRay<float>;
    using Ray4 = TRay<Array<float, 4>>;
    using Ray8 = TRay<Array<float, 8>>;
    using Ray16 = TRay<Array<float, 16>>;
    template <typename Value> struct TVertex {
        using Vector3 = vec<3, Value>;
        using Vector2 = vec<2, Value>;
        Vector3 pos, Ns;
        Vector2 texCoord;
        TVertex() = default;
    };
    using Vertex = TVertex<float>;
    template <typename Value> struct TSurfaceSample {
        using Vector3 = vec<3, Value>;
        using Vector2 = vec<2, Value>;
        Vector2 uv;
        Value pdf;
        Vector3 p;
        Vector3 normal;
    };
    using SurfaceSample = TSurfaceSample<float>;
    template <typename Value, typename IValue> struct TIntersection;
    template <typename Value> struct TTriangle {
        using Vector3 = vec<3, Value>;
        using Vector2 = vec<2, Value>;
        std::array<Vector3, 3> v;
        std::array<Vector2, 3> texCoords;
        std::array<Vector3, 3> Ns;
        vec3 Ng;
        [[nodiscard]] Vector3 interpolated_normal(const Vector2 &uv) const {
            return normalize(Interpolate(Ns[0], Ns[1], Ns[2], uv));
        }
        [[nodiscard]] Vector2 interpolated_tex_coord(const Vector2 &uv) const {
            return Interpolate(texCoords[0], texCoords[1], texCoords[2], uv);
        }
        [[nodiscard]] Value Area() const {
            Vector3 e1 = (v[1] - v[0]);
            Vector3 e2 = (v[2] - v[0]);
            return length(cross(e1, e2)) * 0.5f;
        }
        void sample_surface(Vector2 u, SurfaceSample *sample) const {
            Value su0 = std::sqrt(u[0]);
            Vector2 b = Vector2(1 - su0, u[1] * su0);

            sample->uv = u;
            sample->p = Interpolate(v[0], v[1], v[2], b);
            sample->pdf = 1.0f / Area();
            sample->normal = interpolated_normal(u);
        }

        template <typename IValue> bool Intersect(const TRay<Value> &ray, TIntersection<Value, IValue> *) const;
    };
    using Triangle = TTriangle<float>;
    struct PrimitiveHandle {
        int32_t meshId = -1;
        int32_t primId = -1;
    };

    template <typename Value, typename IValue> struct TIntersection {
        Value t = Inf;
        using Vector3 = vec<3, Value>;
        using Vector2 = vec<2, Value>;
        TTriangle<Value> triangle;
        IValue meshId = -1;
        IValue primId = -1;
        Vector2 uv;
        Vector3 Ng;
        Vector3 p;
        Vector3 wo;
        TIntersection() = default;
        explicit TIntersection(const TRay<Value> &ray) : wo(-ray.d) {}
    };
    using Intersection = TIntersection<float, int32_t>;
    template <typename Value>
    template <typename IValue>
    inline bool TTriangle<Value>::Intersect(const TRay<Value> &ray, TIntersection<Value, IValue> *intersection) const {
        auto &v0 = v[0];
        auto &v1 = v[1];
        auto &v2 = v[2];
        Vector3 e1 = (v1 - v0);
        Vector3 e2 = (v2 - v0);
        float a, f, u, _v;
        auto h = cross(ray.d, e2);
        a = dot(e1, h);
        if (a > -1e-6f && a < 1e-6f)
            return false;
        f = 1.0f / a;
        auto s = ray.o - v0;
        u = f * dot(s, h);
        if (u < 0.0 || u > 1.0)
            return false;
        auto q = cross(s, e1);
        _v = f * dot(ray.d, q);
        if (_v < 0.0 || u + _v > 1.0)
            return false;
        float t = f * dot(e2, q);
        if (t > ray.t_min && t < ray.t_max) {
            intersection->t = t;
            intersection->p = ray.o + t * ray.d;
            intersection->Ng = Ng;
            intersection->uv = Vector2(u, _v);
            return true;
        } else {
            return false;
        }
    }
    class BSDF;

    template <typename Value> struct TShadingPoint {
        using Vector2 = vec<2, Value>;
        Vector2 texCoords;
    };
    using ShadingPoint = TShadingPoint<float>;
    enum TransportMode : uint32_t { EImportance, ERadiance };

} // namespace akari
#endif // AKARIRENDER_GEOMETRY_HPP
