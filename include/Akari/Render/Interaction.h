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

#ifndef AKARIRENDER_INTERACTION_H
#define AKARIRENDER_INTERACTION_H

#include <Akari/Render/Geometry.hpp>

namespace Akari{
    struct Interaction {
        vec3 wo; // world space wo
        vec3 p;
        vec3 Ng{};
        Interaction()= default;
        Interaction(const vec3 &wo, const vec3 &p, const vec3 &Ng) : wo(wo), p(p), Ng(Ng) {}

        [[nodiscard]] Ray SpawnRay(const vec3 &w, Float rayBias = Eps) const {
            return Ray(p, w, rayBias / abs(dot(w, Ng)), Inf);
        }
        [[nodiscard]] Ray SpawnTo(const vec3 &q, Float rayBias = Eps) const {
            auto w = (q - p);
            return Ray(p, w, rayBias / abs(dot(w, Ng)) / length(w), 1 - ShadowEps);
        }
    };
    class MemoryArena;

    struct AKR_EXPORT SurfaceInteraction : Interaction {
        ShadingPoint sp{};
        vec3 Ns{};
        BSDF *bsdf = nullptr;
        SurfaceInteraction(const vec3 &wo, const vec3 &p, const Triangle &triangle, const Intersection &intersection)
            : Interaction(wo, p, triangle.Ng) {
            sp.texCoords = triangle.InterpolatedTexCoord(intersection.uv);
            Ns = triangle.InterpolatedNormal(intersection.uv);
            Ng = triangle.Ng;
        }
        SurfaceInteraction(const vec3 &wo, const vec3 &p, const Triangle &triangle, const Intersection &intersection, MemoryArena &arena);
    };
    class EndPoint;
    struct AKR_EXPORT EndPointInteraction : Interaction {
        EndPoint * ep = nullptr;
    };
}



#endif // AKARIRENDER_INTERACTION_H
