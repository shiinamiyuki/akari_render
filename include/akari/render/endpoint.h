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

#ifndef AKARIRENDER_ENDPOINT_H
#define AKARIRENDER_ENDPOINT_H

#include <akari/core/component.h>
#include <akari/core/spectrum.h>
#include <akari/render/geometry.hpp>
#include <akari/render/interaction.h>
#include <akari/render/scene.h>
namespace akari {
    template <typename Float, typename Spectrum> struct RayIncidentSample {
        AKR_BASIC_TYPES()
        Vector3f wi;
        Spectrum I;
        Vector3f normal;
        Vector2i pos; // the uv coordinate of the sampled position or the raster position (0,0) to (bound.x, boubd.y)
        Float pdf;
    };
    template <typename Float, typename Spectrum> struct RayEmissionSample {
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(Ray)
        Ray ray;
        Spectrum E;
        Vector3f normal;
        Vector2f uv; // 2D parameterized position
        Float pdfPos, pdfDir;
    };
    template <typename Float, typename Spectrum> struct VisibilityTester {
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(Ray, Interaction)
        Ray shadowRay{};
        VisibilityTester() = default;
        VisibilityTester(const Interaction &p1, const Interaction &p2) {
            auto w = p2.p - p1.p;
            auto dist = length(w);
            w /= dist;
            shadowRay = Ray(p1.p, w, Eps() / abs(dot(w, p1.Ng)), dist * (1.0 - ShadowEps()));
        }
        [[nodiscard]] Bool visible(const Scene &scene) const { return !scene.Occlude(shadowRay); }
        [[nodiscard]] Spectrum Tr(const Scene &scene) const {
            return scene.Occlude(shadowRay) ? Spectrum(0) : Spectrum(1);
        }
    };
    template <typename Float, typename Spectrum> class EndPoint : public Component {
      public:
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(Ray, Interaction, VisibilityTester, RayEmissionSample, RayIncidentSample)
        virtual Float pdf_incidence(const Interaction &ref, const Vector3f &wi) const = 0;
        virtual void pdf_emission(const Ray &ray, Float *pdfPos, Float *pdfDir) const = 0;
        virtual void sample_incidence(const Vector2i &u, const Interaction &ref, RayIncidentSample *sample,
                                      VisibilityTester *tester) const = 0;
        virtual void sample_emission(const vec2 &u1, const vec2 &u2, RayEmissionSample *sample) const = 0;
    };
} // namespace akari
#endif // AKARIRENDER_ENDPOINT_H
