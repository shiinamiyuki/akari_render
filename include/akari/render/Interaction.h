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

#include <akari/render/geometry.hpp>
#include <akari/render/medium_stack.h>
namespace akari {
    template<typename Float, typename Spectrum>
    struct Interaction {
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(Ray)
        Vector3f wo = Vector3f(0); // world space wo
        Vector3f p = Vector3f(0);
        Vector3f Ng = Vector3f(0);
        Vector2f uv{};
        Interaction() = default;
        Interaction(const Vector3f &wo, const Vector3f &p, const Vector3f &Ng) : wo(wo), p(p), Ng(Ng) {}

        [[nodiscard]] Ray spawn_dir(const Vector3f &w, Float rayBias = Eps()) const {
            return Ray(p, w, rayBias / abs(dot(w, Ng)), Inf);
        }
        [[nodiscard]] Ray spawn_to(const Vector3f &q, Float rayBias = Eps()) const {
            auto w = (q - p);
            return Ray(p, w, rayBias / abs(dot(w, Ng)) / length(w), 1 - ShadowEps());
        }
    };
    class MemoryArena;
    class Scene;
    template<typename Float, typename Spectrum>
    struct MaterialSlot;
    template<typename Float, typename Spectrum>
    struct AKR_EXPORT SurfaceInteraction : Interaction<Float, Spectrum> {
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(Interaction, ShadingPoint)
        using Interaction::Interaction;
        ShadingPoint sp{};
        Vector3f Ns = Vector3f(0);
        BSDF *bsdf = nullptr;
        PrimitiveHandle handle;
        MediumStack *mediumStack = nullptr;
        const MaterialSlot *materialSlot = nullptr;
        SurfaceInteraction(const MaterialSlot *materialSlot, const Vector3f &wo, const Vector3f &p, const Triangle &triangle,
                           const Intersection &intersection)
            : Interaction(wo, p, triangle.Ng), materialSlot(materialSlot) {
            sp.texCoords = triangle.interpolated_tex_coord(intersection.uv);
            uv = intersection.uv;
            Ns = triangle.interpolated_normal(intersection.uv);
            handle.meshId = intersection.meshId;
            handle.primId = intersection.primId;
        }
        SurfaceInteraction(const MaterialSlot *materialSlot, const Vector3f &wo, const Vector3f &p, const Triangle &triangle,
                           const Intersection &intersection, MemoryArena &arena);
        void compute_scattering_functions(MemoryArena &, TransportMode mode, float scale);
        Spectrum Le(const Vector3f &wo);
    };
    class EndPoint;
    struct AKR_EXPORT EndPointInteraction : Interaction {
        using Interaction::Interaction;
        const EndPoint *ep = nullptr;
        EndPointInteraction(const EndPoint *ep, const Vector3f &p, const Vector3f &Ng) : Interaction(Vector3f(0), p, Ng), ep(ep) {}
    };
} // namespace akari

#endif // AKARIRENDER_INTERACTION_H
