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

#ifndef AKARIRENDER_LIGHT_H
#define AKARIRENDER_LIGHT_H

#include <Akari/Core/Component.h>
#include <Akari/Render/Geometry.hpp>
#include <Akari/Render/Scene.h>
#include <Akari/Core/Spectrum.h>

namespace Akari {
    struct LightSample {
        vec3 wi;
        Spectrum Li;
        vec3 normal;
        float pdf;
    };

    struct LightRaySample {
        Ray ray;
        Spectrum Le;
        vec3 normal;
        float pdfPos, pdfDir;
    };
    struct VisibilityTester {
        Ray shadowRay;
        bool visible(const Scene &scene) const { return !scene.Occlude(shadowRay); }
    };
    class Light : public Component {
      public:
        virtual Float Power() const = 0;
        virtual Spectrum Li(const vec3 &wo, ShadingPoint &sp) const = 0;

        virtual void SampleLi(const vec2 &u, Intersection &isct, LightSample &sample, VisibilityTester &) const = 0;

        virtual Float PdfLi(const Intersection &intersection, const vec3 &wi) const = 0;

        //        virtual void SampleLe(const Point2f &u1, const Point2f &u2, LightRaySample &sample) = 0;
    };

} // namespace Akari

#endif // AKARIRENDER_LIGHT_H
