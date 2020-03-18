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

#ifndef AKARIRENDER_REFLECTION_HPP
#define AKARIRENDER_REFLECTION_HPP

#include <Akari/Render/BSDF.h>
namespace Akari {
    class LambertianReflection : public BSDFComponent {
        Spectrum R;

      public:
        explicit LambertianReflection(const Spectrum &R)
            : BSDFComponent(BSDFType(BSDF_DIFFUSE | BSDF_REFLECTION)), R(R) {}
        [[nodiscard]] Spectrum Evaluate(const vec3 &wo, const vec3 &wi) const override { return Akari::Spectrum(); }
    };
    class SpecularReflection : public BSDFComponent {
        Spectrum R;

      public:
        explicit SpecularReflection(const Spectrum &R)
            : BSDFComponent(BSDFType(BSDF_SPECULAR | BSDF_REFLECTION)), R(R) {}
        [[nodiscard]] Float EvaluatePdf(const vec3 &wo, const vec3 &wi) const override { return 0; }
        [[nodiscard]] Spectrum Evaluate(const vec3 &wo, const vec3 &wi) const override { return Spectrum(0); }
        void Sample(BSDFSample &sample) const override {
            sample.wi = Reflect(sample.wi, vec3(0, 1, 0));
            sample.f = R / AbsCosTheta(sample.wi);
            sample.pdf = 1;
        }
    };
} // namespace Akari

#endif // AKARIRENDER_REFLECTION_HPP
