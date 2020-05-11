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
#include "reflection.h"
#include <akari/render/bsdf.h>
namespace akari {
    enum MicrofacetType {
        EGGX,
        EBeckmann,
        EPhong,

    };

    template <typename Float, typename Spectrum> struct Microfacet {
        AKR_BASIC_TYPES()
        static inline Float BeckmannD(Float alpha, const Vector3f &m) {
            if (m.y <= 0.0f)
                return 0.0f;
            auto c = Cos2Theta(m);
            auto t = Tan2Theta(m);
            auto a2 = alpha * alpha;
            return std::exp(-t / a2) / (Pi * a2 * c * c);
        }

        static inline Float BeckmannG1(Float alpha, const Vector3f &v, const Vector3f &m) {
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
        static inline Float PhongG1(Float alpha, const Vector3f &v, const Vector3f &m) {
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

        static inline Float PhongD(Float alpha, const Vector3f &m) {
            if (m.y <= 0.0f)
                return 0.0f;
            return (alpha + 2) / (2 * Pi) * std::pow(m.y, alpha);
        }

        static inline Float GGX_D(Float alpha, const Vector3f &m) {
            if (m.y <= 0.0f)
                return 0.0f;
            Float a2 = alpha * alpha;
            auto  c2 = Cos2Theta(m);
            auto  t2 = Tan2Theta(m);
            auto  at = (a2 + t2);
            return a2 / (Pi * c2 * c2 * at * at);
        }

        static inline Float GGX_G1(Float alpha, const Vector3f &v, const Vector3f &m) {
            if (dot(v, m) * v.y <= 0) {
                return 0.0f;
            }
            return 2.0 / (1.0 + std::sqrt(1.0 + alpha * alpha * Tan2Theta(m)));
        }
    };
    // see https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    template <typename Float, typename Spectrum> struct MicrofacetModel {
        AKR_BASIC_TYPES()
        MicrofacetModel(MicrofacetType type, Float roughness) : type(type) {
            if (type == EPhong) {
                alpha = 2.0f / (roughness * roughness) - 2.0f;
            } else {
                alpha = roughness;
            }
        }
        [[nodiscard]] Float D(const Vector3f &m) const {
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
        [[nodiscard]] Float G1(const Vector3f &v, const Vector3f &m) const {
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
        [[nodiscard]] Float G(const Vector3f &i, const Vector3f &o, const Vector3f &m) const {
            return G1(i, m) * G1(o, m);
        }
        [[nodiscard]] Vector3f sample_wh(const Vector3f &wo, const vec2 &u) const {
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
            auto wh = Vector3f(std::cos(phi) * sinTheta, cosTheta, std::sin(phi) * sinTheta);
            if (!same_hemisphere(wo, wh))
                wh = -wh;
            return wh;
        }
        [[nodiscard]] Float evaluate_pdf(const Vector3f &wh) const { return D(wh) * abs_cos_theta(wh); }

      private:
        MicrofacetType type;
        Float          alpha;
    };
    template <typename Float, typename Spectrum>
    class AKR_EXPORT MicrofacetReflection : public BSDFComponent<Float, Spectrum> {
        const Spectrum R;
        using MicrofacetModel = akari::MicrofacetModel<Float, Spectrum>;
        using Fresnel = akari::Fresnel<Float, Spectrum>;
        const MicrofacetModel microfacet;
        const Fresnel *       fresnel;

      public:
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(BSDFComponent)
        MicrofacetReflection(const Vector3f &R, MicrofacetModel microfacet, const Fresnel *fresnel)
            : BSDFComponent(BSDFType(BSDF_REFLECTION | BSDF_GLOSSY)), R(R), microfacet(microfacet), fresnel(fresnel) {}
        [[nodiscard]] Float    evaluate_pdf(const Vector3f &wo, const Vector3f &wi) const override;
        [[nodiscard]] Spectrum evaluate(const Vector3f &wo, const Vector3f &wi) const override;
        Spectrum               sample(const vec2 &u, const Vector3f &wo, Vector3f *wi, Float *pdf,
                                      BSDFType *sampledType) const override;
    };
    template <typename Float, typename Spectrum>
    class AKR_EXPORT MicrofacetTransmission : public BSDFComponent<Float, Spectrum> {
        const Spectrum T;
        using MicrofacetModel = akari::MicrofacetModel<Float, Spectrum>;
        using FresnelDielectric = akari::FresnelDielectric<Float, Spectrum>;
        const MicrofacetModel   microfacet;
        const FresnelDielectric fresnel;
        const Float             etaA, etaB;
        TransportMode           mode;

      public:
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(BSDFComponent)
        MicrofacetTransmission(const Vector3f &T, MicrofacetModel microfacet, Float etaA, Float etaB,
                               TransportMode mode)
            : BSDFComponent(BSDFType(BSDF_TRANSMISSION | BSDF_GLOSSY)), T(T), microfacet(microfacet),
              fresnel(etaA, etaB), etaA(etaA), etaB(etaB), mode(mode) {}
        [[nodiscard]] Float    evaluate_pdf(const Vector3f &wo, const Vector3f &wi) const override;
        [[nodiscard]] Spectrum evaluate(const Vector3f &wo, const Vector3f &wi) const override;
        Spectrum               sample(const vec2 &u, const Vector3f &wo, Vector3f *wi, Float *pdf,
                                      BSDFType *sampledType) const override;
    };

} // namespace akari
#endif // AKARIRENDER_MICROFACET_H
