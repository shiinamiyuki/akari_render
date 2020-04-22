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

#include <Akari/Render/Material.h>

namespace Akari {
    inline Float SchlickWeight(Float cosTheta) { return Power<5>(std::clamp<float>(1 - cosTheta, 0, 1)); }
    inline Float FrSchlick(Float R0, Float cosTheta) { return lerp(SchlickWeight(cosTheta), R0, 1.0f); }
    inline Spectrum FrSchlick(const Spectrum &R0, Float cosTheta) {
        return lerp(Spectrum(SchlickWeight(cosTheta)), R0, Spectrum(1));
    }
    class DisneyDiffuse : public BSDFComponent {
        Spectrum R;

      public:
        DisneyDiffuse(const Spectrum &R) : BSDFComponent(BSDFType(BSDF_DIFFUSE | BSDF_REFLECTION)), R(R) {}
        [[nodiscard]] Spectrum Evaluate(const vec3 &wo, const vec3 &wi) const override {
            Float Fo = SchlickWeight(AbsCosTheta(wo));
            Float Fi = SchlickWeight(AbsCosTheta(wi));
            return R * InvPi * (1 - Fo * 0.5) * (1 - Fi * 0.5);
        }
    };

    inline Spectrum CalculateTint(const Spectrum &baseColor) {
        auto luminance = baseColor.Luminance();
        return luminance > 0 ? baseColor / luminance : Spectrum(1);
    }

    class DisneySheen : public BSDFComponent {
        const Spectrum R;
        const Spectrum sheen;
        const Spectrum sheenTint;

      public:
        DisneySheen(const Spectrum &R, const Spectrum &sheen, const Spectrum &sheenTint)
            : BSDFComponent(BSDFType(BSDF_GLOSSY | BSDF_REFLECTION)), R(R), sheen(sheen), sheenTint(sheenTint) {}
        [[nodiscard]] Spectrum Evaluate(const vec3 &wo, const vec3 &wi) const override {
            auto wh = wi + wo;
            if (all(equal(wh, vec3(0)))) {
                return Spectrum(0);
            }
            wh = normalize(wh);
            Float d = dot(wh, wi);
            return sheen * lerp(Spectrum(1), CalculateTint(R), sheenTint) * SchlickWeight(d);
        }
    };

    static inline Float D_GTR1(Float cosThetaH, Float alpha) {
        if (alpha >= 1)
            return InvPi;
        auto a2 = alpha * alpha;
        return (a2 - 1) / (Pi * log(a2)) * (1.0f / (1 + (a2 - 1) * cosThetaH * cosThetaH));
    }
    static inline Float SmithGGX_G1(Float cosTheta, Float alpha) {
        auto a2 = alpha * alpha;
        return 1.0 / (cosTheta + std::sqrt(a2 + cosTheta - a2 * cosTheta * cosTheta));
    }
    class DisneyClearCoat : public BSDFComponent {
        Float clearcoat;
        Float alpha;

      public:
        DisneyClearCoat(Float clearcoat, Float alpha)
            : BSDFComponent(BSDFType(BSDF_GLOSSY | BSDF_REFLECTION)), clearcoat(clearcoat), alpha(alpha) {}
        [[nodiscard]] Spectrum Evaluate(const vec3 &wo, const vec3 &wi) const override {
            auto wh = wi + wo;
            if (all(equal(wh, vec3(0)))) {
                return Spectrum(0);
            }
            wh = normalize(wh);
            Float D = D_GTR1(AbsCosTheta(wh), alpha);
            Float F = FrSchlick(0.04, dot(wo, wh));
            Float G = SmithGGX_G1(AbsCosTheta(wi), 0.25) * SmithGGX_G1(AbsCosTheta(wo), 0.25);
            return Spectrum(0.25 * clearcoat * D * F * G);
        }
    };
} // namespace Akari
