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

#include <akari/core/component.h>
#include <akari/render/types.hpp>

namespace akari {
    template <typename Float, typename Spectrum> struct Ray {
        AKR_BASIC_TYPES()
        Vector3f o, d;
        Float t_min, t_max;

        Ray() = default;

        Ray(const Vector3f &o, const Vector3f &d, Float t_min = Eps(),
            Float t_max = std::numeric_limits<float>::infinity())
            : o(o), d(d), t_min(t_min), t_max(t_max) {}

        [[nodiscard]] Vector3f At(Float t) const { return o + t * d; }
    };

#define AKR_IMPORT_TYPES() using Ray = Ray<Float>;

    template <typename Float, typename Spectrum> struct Vertex {
        AKR_BASIC_TYPES()
        Vector3f pos, Ns;
        Vector2f texCoord;
        Vertex() = default;
    };
    template <typename Float, typename Spectrum> struct SurfaceSample {
        AKR_BASIC_TYPES()
        Vector2f uv;
        Float pdf;
        Vector3f p;
        Vector3f normal;
    };
    template <typename Float, typename Spectrum> struct Triangle {
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(Ray, Intersection, SurfaceSample)
        std::array<Vector3f, 3> v;
        std::array<Vector2f, 3> texCoords;
        std::array<Vector3f, 3> Ns;
        Vector3f Ng;
        [[nodiscard]] Vector3f interpolated_normal(const Vector2f &uv) const {
            return normalize(Interpolate(Ns[0], Ns[1], Ns[2], uv));
        }
        [[nodiscard]] Vector2f interpolated_tex_coord(const Vector2f &uv) const {
            return Interpolate(texCoords[0], texCoords[1], texCoords[2], uv);
        }
        [[nodiscard]] Float Area() const {
            Vector3f e1 = (v[1] - v[0]);
            Vector3f e2 = (v[2] - v[0]);
            return length(cross(e1, e2)) * 0.5f;
        }
        void sample_surface(Vector2f u, SurfaceSample *sample) const {
            Float su0 = std::sqrt(u[0]);
            Vector2f b = Vector2f(1 - su0, u[1] * su0);

            sample->uv = u;
            sample->p = Interpolate(v[0], v[1], v[2], b);
            sample->pdf = 1.0f / Area();
            sample->normal = interpolated_normal(u);
        }

        bool Intersect(const Ray &ray, Intersection *) const;
    };
    struct PrimitiveHandle {
        int32_t meshId = -1;
        int32_t primId = -1;
    };

    template <typename Float, typename Spectrum> struct Intersection {
        Float t = Inf;
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(Triangle, Ray)
        Triangle triangle;
        Int meshId = -1;
        Int primId = -1;
        Vector2f uv;
        Vector3f Ng;
        Vector3f p;
        Vector3f wo;
        Intersection() = default;
        explicit Intersection(const Ray &ray) : wo(-ray.d) {}
    };

    template <typename Float, typename Spectrum>
    inline bool Triangle<Float, Spectrum>::Intersect(const Ray &ray, Intersection *intersection) const {
        auto &v0 = v[0];
        auto &v1 = v[1];
        auto &v2 = v[2];
        Vector3f e1 = (v1 - v0);
        Vector3f e2 = (v2 - v0);
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
            intersection->uv = Vector2f(u, _v);
            return true;
        } else {
            return false;
        }
    }

    template <typename Float, typename Spectrum> struct ShadingPoint {
        AKR_BASIC_TYPES()
        Vector2f texCoords;
    };

    enum TransportMode : uint32_t { EImportance, ERadiance };

} // namespace akari
#endif // AKARIRENDER_GEOMETRY_HPP
