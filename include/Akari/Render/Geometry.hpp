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

#include <Akari/Core/Component.h>
#include <Akari/Core/Math.h>

namespace Akari {
    struct Ray {
        vec3 o, d;
        float t_min, t_max;

        Ray() = default;

        Ray(const vec3 &o, const vec3 &d, float t_min = Eps(), float t_max = std::numeric_limits<float>::infinity())
            : o(o), d(d), t_min(t_min), t_max(t_max) {}

        [[nodiscard]] vec3 At(Float t) const { return o + t * d; }
    };

    struct Vertex {
        vec3 pos, Ns;
        vec2 texCoord;
        Vertex() = default;
    };
    struct SurfaceSample {
        vec2 uv;
        Float pdf;
        vec3 p;
        vec3 normal;
    };
    struct Intersection;
    struct Triangle {
        std::array<vec3, 3> v;
        std::array<vec2, 3> texCoords;
        std::array<vec3, 3> Ns;
        vec3 Ng;
        [[nodiscard]] vec3 InterpolatedNormal(const vec2 &uv) const { return Interpolate(Ns[0], Ns[1], Ns[2], uv); }
        [[nodiscard]] vec2 InterpolatedTexCoord(const vec2 &uv) const {
            return Interpolate(texCoords[0], texCoords[1], texCoords[2], uv);
        }
        [[nodiscard]] Float Area() const {
            vec3 e1 = (v[1] - v[0]);
            vec3 e2 = (v[2] - v[0]);
            return length(cross(e1, e2)) * 0.5f;
        }
        void Sample(vec2 u, SurfaceSample *sample) const {
            if (u.x + u.y > 1) {
                u.x = 1 - u.x;
                u.y = 1 - u.y;
            }
            sample->uv = u;
            sample->p = Interpolate(v[0], v[1], v[2], u);
            sample->pdf = 1.0f / Area();
            sample->normal = InterpolatedNormal(u);
        }

        bool Intersect(const Ray &ray, Intersection *) const;
    };
    struct PrimitiveHandle{
        int32_t meshId = -1;
        int32_t primId = -1;
    };
    struct Intersection {
        Float t = Inf;
        Triangle triangle;
        int32_t meshId = -1;
        int32_t primId = -1;
        vec2 uv;
        vec3 Ng;
        vec3 p;
        vec3 wo;
        Intersection() = default;
        explicit Intersection(const Ray &ray) : wo(-ray.d) {}
    };
    inline bool Triangle::Intersect(const Ray &ray, Intersection *intersection) const {
        auto &v0 = v[0];
        auto &v1 = v[1];
        auto &v2 = v[2];
        vec3 e1 = (v1 - v0);
        vec3 e2 = (v2 - v0);
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
            intersection->uv = vec2(u, _v);
            return true;
        } else {
            return false;
        }
    }
    class BSDF;

    struct ShadingPoint {
        vec2 texCoords;
    };

    enum TransportMode : uint32_t { EImportance, ERadiance };



} // namespace Akari
#endif // AKARIRENDER_GEOMETRY_HPP
