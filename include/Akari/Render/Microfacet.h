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

#ifndef AKARIRENDER_MICROFACET_H
#define AKARIRENDER_MICROFACET_H
#include "Reflection.h"
#include <Akari/Render/BSDF.h>
namespace Akari {
    enum MicrofacetType {
        EGGX,
        EBeckmann,
        EPhong,

    };

    inline Float BeckmannD(Float alpha, const vec3 &m) {
        if (m.y <= 0.0f)
            return 0.0f;
        auto c = Cos2Theta(m);
        auto t = Tan2Theta(m);
        auto a2 = alpha * alpha;
        return std::exp(-t / a2) / (Pi * a2 * c * c);
    }

    inline Float BeckmannG1(Float alpha, const vec3 &v, const vec3 &m) {
        if (dot(v, m) * v.y <= 0) {
            return 0.0f;
        }
        auto a = 1.0f / (alpha * TanTheta(v));
        if (a < 1.6) {
            return (3.535 * a + 2.181 * a * a) / (1.0f + 2.276 * a + 2.577 * a * a);
        } else {
            return 1.0f;
        }
    }
    inline Float PhongG1(Float alpha, const vec3 &v, const vec3 &m) {
        if (dot(v, m) * v.y <= 0) {
            return 0.0f;
        }
        auto a = std::sqrt(0.5f * alpha + 1.0f) / TanTheta(v);
        if (a < 1.6) {
            return (3.535 * a + 2.181 * a * a) / (1.0f + 2.276 * a + 2.577 * a * a);
        } else {
            return 1.0f;
        }
    }

    inline Float PhongD(Float alpha, const vec3 &m) {
        if (m.y <= 0.0f)
            return 0.0f;
        return (alpha + 2) / (2 * Pi) * std::pow(m.y, alpha);
    }

    inline Float GGX_D(Float alpha, const vec3 &m) {
        if (m.y <= 0.0f)
            return 0.0f;
        Float a2 = alpha * alpha;
        auto c2 = Cos2Theta(m);
        auto t2 = Tan2Theta(m);
        auto at = (a2 + t2);
        return a2 / (Pi * c2 * c2 * at * at);
    }

    inline Float GGX_G1(Float alpha, const vec3 &v, const vec3 &m) {
        if (dot(v, m) * v.y <= 0) {
            return 0.0f;
        }
        return 2.0 / (1.0 + std::sqrt(1.0 + alpha * alpha * Tan2Theta(m)));
    }
    // see https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    struct MicrofacetModel {
        MicrofacetModel(MicrofacetType type, Float roughness) : type(type) {
            if (type == EPhong) {
                alpha = 2.0f / (roughness * roughness) - 2.0f;
            } else {
                alpha = roughness;
            }
        }
        [[nodiscard]] Float D(const vec3 &m) const {
            switch (type) {
            case EBeckmann:
                return BeckmannD(alpha, m);
            case EPhong:
                return PhongD(alpha, m);
            case EGGX:
                return GGX_D(alpha, m);
            }

            return 0.0f;
        }
        [[nodiscard]] Float G1(const vec3 &v, const vec3 &m) const {
            switch (type) {
            case EBeckmann:
                return BeckmannG1(alpha, v, m);
            case EPhong:
                return PhongG1(alpha, v, m);
            case EGGX:
                return GGX_G1(alpha, v, m);
            }
            return 0.0f;
        }
        [[nodiscard]] Float G(const vec3 &i, const vec3 &o, const vec3 &m) const { return G1(i, m) * G1(o, m); }
        [[nodiscard]] vec3 sampleWh(const vec3 &wo, const vec2 &u) const {
            Float phi = 2 * Pi * u[1];
            Float cosTheta = 0;
            switch (type) {
            case EBeckmann: {
                Float t2 = -alpha * alpha * std::log(1 - u[0]);
                cosTheta = 1.0f / std::sqrt(1 + t2);
                break;
            }
            case EPhong: {
                cosTheta = std::pow((double)u[0], 1.0 / ((double)alpha + 2.0f));
                break;
            }
            case EGGX: {
                Float t2 = alpha * alpha * u[0] / (1 - u[0]);
                cosTheta = 1.0f / std::sqrt(1 + t2);
                break;
            }
            }
            auto sinTheta = std::sqrt(std::max(0.0f, 1 - cosTheta * cosTheta));
            auto wh = vec3(std::cos(phi) * sinTheta, cosTheta, std::sin(phi) * sinTheta);
            if (!SameHemisphere(wo, wh))
                wh = -wh;
            return wh;
        }
        [[nodiscard]] Float evaluatePdf(const vec3 &wh) const { return D(wh) * AbsCosTheta(wh); }

      private:
        MicrofacetType type;
        Float alpha;
    };

    class AKR_EXPORT MicrofacetReflection : public BSDFComponent {
        const Spectrum R;
        const MicrofacetModel microfacet;
        const Fresnel *fresnel;

      public:
        MicrofacetReflection(const vec3 &R, MicrofacetModel microfacet, const Fresnel *fresnel)
            : BSDFComponent(BSDFType(BSDF_REFLECTION | BSDF_GLOSSY)), R(R), microfacet(microfacet), fresnel(fresnel) {}
        [[nodiscard]] Float EvaluatePdf(const vec3 &wo, const vec3 &wi) const override;
        [[nodiscard]] Spectrum Evaluate(const vec3 &wo, const vec3 &wi) const override;
        Spectrum Sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf, BSDFType *sampledType) const override;
    };
    class AKR_EXPORT MicrofacetTransmission : public BSDFComponent {
        const Spectrum T;
        const MicrofacetModel microfacet;
        const FresnelDielectric fresnel;
        const Float etaA, etaB;
        TransportMode mode;

      public:
        MicrofacetTransmission(const vec3 &T, MicrofacetModel microfacet, Float etaA, Float etaB, TransportMode mode)
            : BSDFComponent(BSDFType(BSDF_TRANSMISSION | BSDF_GLOSSY)), T(T), microfacet(microfacet),
              fresnel(etaA, etaB), etaA(etaA), etaB(etaB), mode(mode) {}
        [[nodiscard]] Float EvaluatePdf(const vec3 &wo, const vec3 &wi) const override;
        [[nodiscard]] Spectrum Evaluate(const vec3 &wo, const vec3 &wi) const override;
        Spectrum Sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf, BSDFType *sampledType) const override;
    };

} // namespace Akari
#endif // AKARIRENDER_MICROFACET_H
