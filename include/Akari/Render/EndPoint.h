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

#include <Akari/Core/Component.h>
#include <Akari/Render/Geometry.hpp>
#include <Akari/Render/Scene.h>
#include <Akari/Render/Interaction.h>
namespace Akari {
    struct RayIncidentSample {
        vec3 wi;
        Spectrum I;
        vec3 normal;
        float pdf;
    };
    struct RayEmissionSample {
        Ray ray;
        Spectrum E;
        vec3 normal;
        float pdfPos, pdfDir;
    };
    struct VisibilityTester {
        Ray shadowRay;
        [[nodiscard]] bool visible(const Scene &scene) const { return !scene.Occlude(shadowRay); }
    };

    class EndPoint : public Component {
      public:
        virtual Float PdfIncidence(const Interaction &ref, const vec3 &wi) const = 0;
        virtual void PdfEmission(const Ray &ray, Float *pdfPos, Float *pdfDir) const = 0;
        virtual void SampleIncidence(const vec2 &u, const Interaction &ref, RayIncidentSample *sample,
                                     VisibilityTester *tester) const = 0;
        virtual void SampleEmission(const vec2 &u1, const vec2 &u2, RayEmissionSample *sample) const = 0;
    };
} // namespace Akari
#endif // AKARIRENDER_ENDPOINT_H
