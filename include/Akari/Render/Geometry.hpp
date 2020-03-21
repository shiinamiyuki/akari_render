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

        Ray(const vec3 &o, const vec3 &d, float t_min, float t_max = std::numeric_limits<float>::infinity())
            : o(o), d(d), t_min(t_min), t_max(t_max) {}

        [[nodiscard]] vec3 At(Float t) const { return o + t * d; }
    };

    struct Vertex {
        vec3 pos, Ns;
        vec2 texCoord;
        Vertex() = default;
    };
    struct Triangle {
        std::array<vec3, 3> v;
        std::array<vec2, 3> texCoords;
        std::array<vec3, 3> Ns;
        vec3 Ng;
        [[nodiscard]] vec3 InterpolatedNormal(const vec2 &uv) const { return Interpolate(Ns[0], Ns[1], Ns[2], uv); }
        [[nodiscard]] vec2 InterpolatedTexCoord(const vec2 &uv) const {
            return Interpolate(texCoords[0], texCoords[1], texCoords[2], uv);
        }
    };

    struct Intersection {
        Float t = Inf;
        Triangle triangle;
        int32_t meshId = -1;
        int32_t primId = -1;
        vec2 uv;
        vec3 Ng;
    };
    class BSDF;

    struct ShadingPoint {
        vec2 texCoords;
    };

    struct ScatteringEvent {
        vec3 wo; // world space wo
        vec3 p;
        vec3 Ns{};
        vec3 Ng{};
        ShadingPoint sp{};
        BSDF *bsdf = nullptr;
        Float rayBias = Eps;
        ScatteringEvent(const vec3 &wo, const vec3 &p, const Triangle &triangle, const Intersection &intersection)
            : wo(wo), p(p) {
            sp.texCoords = triangle.InterpolatedTexCoord(intersection.uv);
            Ns = triangle.InterpolatedNormal(intersection.uv);
            Ng = triangle.Ng;
        }

        [[nodiscard]] Ray SpawnRay(const vec3 &w) const { return Ray(p, w, rayBias / abs(dot(w, Ng)), Inf); }
        [[nodiscard]] Ray SpawnTo(const vec3 &q) const {
            auto w = (q - p);
            return Ray(p, w, rayBias / abs(dot(w, Ng)) / length(w), 1 - ShadowEps);
        }
    };

} // namespace Akari
#endif // AKARIRENDER_GEOMETRY_HPP
