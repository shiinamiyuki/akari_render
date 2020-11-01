

// MIT License
//
// Copyright (c) 2020 椎名深雪
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

#pragma once
#include <akari/core/math.h>
#include <akari/core/variant.h>
#include <akari/render/scenegraph.h>
#include <akari/render/common.h>
#include <akari/core/memory.h>
#include <akari/render/sampler.h>

namespace akari::render {
    enum class BSDFType : int {
        Unset = 0u,
        Reflection = 1u << 0,
        Transmission = 1u << 1,
        Diffuse = 1u << 2,
        Glossy = 1u << 3,
        Specular = 1u << 4,
        DiffuseReflection = Diffuse | Reflection,
        DiffuseTransmission = Diffuse | Transmission,
        GlossyReflection = Glossy | Reflection,
        GlossyTransmission = Glossy | Transmission,
        SpecularReflection = Specular | Reflection,
        SpecularTransmission = Specular | Transmission,
        All = Diffuse | Glossy | Specular | Reflection | Transmission
    };
    inline BSDFType operator&(BSDFType a, BSDFType b) { return BSDFType((int)a & (int)b); }
    inline BSDFType operator|(BSDFType a, BSDFType b) { return BSDFType((int)a | (int)b); }
    inline BSDFType operator~(BSDFType a) { return BSDFType(~(uint32_t)a); }
    class BSDFClosure;
    struct BSDFSample {
        Vec3 wi = Vec3(0);
        Float pdf = 0.0;
        Spectrum f = Spectrum(0.0f);
        BSDFType sampled = BSDFType::Unset;
    };
    class Fresnel {
      public:
        virtual Spectrum evaluate(Float cosThetaI) const = 0;
    };
    class AKR_EXPORT FresnelNoOp : public Fresnel {
      public:
        [[nodiscard]] Spectrum evaluate(Float cosThetaI) const override;
    };

    class AKR_EXPORT FresnelConductor : public Fresnel {
        const Spectrum etaI, etaT, k;

      public:
        FresnelConductor(const Spectrum &etaI, const Spectrum &etaT, const Spectrum &k)
            : etaI(etaI), etaT(etaT), k(k) {}
        [[nodiscard]] Spectrum evaluate(Float cosThetaI) const override;
    };
    class AKR_EXPORT FresnelDielectric : public Fresnel {
        const Float etaI, etaT;

      public:
        FresnelDielectric(const Float &etaI, const Float &etaT) : etaI(etaI), etaT(etaT) {}
        [[nodiscard]] Spectrum evaluate(Float cosThetaI) const override;
    };
    class AKR_EXPORT BSDFClosure {
      public:
        [[nodiscard]] virtual Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const = 0;
        [[nodiscard]] virtual Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const = 0;
        [[nodiscard]] virtual BSDFType type() const = 0;
        [[nodiscard]] virtual bool match_flags(BSDFType flag) const { return ((uint32_t)type() & (uint32_t)flag) != 0; }
        [[nodiscard]] virtual std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const = 0;
    };
    class DiffuseBSDF : public BSDFClosure {
        Spectrum R;

      public:
        DiffuseBSDF(const Spectrum &R) : R(R) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const override {

            if (same_hemisphere(wo, wi)) {
                return cosine_hemisphere_pdf(std::abs(cos_theta(wi)));
            }
            return 0.0f;
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const override {

            if (same_hemisphere(wo, wi)) {
                return R * InvPi;
            }
            return Spectrum(0.0f);
        }
        [[nodiscard]] BSDFType type() const override { return BSDFType::DiffuseReflection; }
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const override {
            BSDFSample sample;
            sample.wi = cosine_hemisphere_sampling(u);
            if (!same_hemisphere(wo, sample.wi)) {
                sample.wi.y = -sample.wi.y;
            }
            sample.sampled = type();
            sample.pdf = cosine_hemisphere_pdf(std::abs(cos_theta(sample.wi)));
            sample.f = R * InvPi;
            return sample;
        }
    };

    class AKR_EXPORT MicrofacetReflection : public BSDFClosure {

        Spectrum R;
        MicrofacetModel model;

      public:
        MicrofacetReflection(const Spectrum &R, Float roughness)
            : R(R), model(microfacet_new(MicrofacetGGX, roughness)) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const override {
            if (same_hemisphere(wo, wi)) {
                auto wh = normalize(wo + wi);
                return microfacet_evaluate_pdf(model, wh) / (Float(4.0f) * dot(wo, wh));
            }
            return 0.0f;
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const override {
            if (same_hemisphere(wo, wi)) {
                Float cosThetaO = abs_cos_theta(wo);
                Float cosThetaI = abs_cos_theta(wi);
                auto wh = (wo + wi);
                if (cosThetaI == 0 || cosThetaO == 0)
                    return Spectrum(0);
                if (wh.x == 0 && wh.y == 0 && wh.z == 0)
                    return Spectrum(0);
                wh = normalize(wh);
                if (wh.y < 0) {
                    wh = -wh;
                }
                auto F = 1.0f; // fresnel->evaluate(dot(wi, wh));

                return R * (microfacet_D(model, wh) * microfacet_G(model, wo, wi, wh) * F /
                            (Float(4.0f) * cosThetaI * cosThetaO));
            }
            return Spectrum(0.0f);
        }
        [[nodiscard]] BSDFType type() const override { return BSDFType::GlossyReflection; }
        [[nodiscard]] std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const override {
            BSDFSample sample;
            sample.sampled = type();
            auto wh = microfacet_sample_wh(model, wo, u);
            sample.wi = glm::reflect(-wo, wh);
            if (!same_hemisphere(wo, sample.wi)) {
                sample.pdf = 0;
                return std::nullopt;
            } else {
                if (wh.y < 0) {
                    wh = -wh;
                }
                sample.pdf = microfacet_evaluate_pdf(model, wh) / (Float(4.0f) * abs(dot(wo, wh)));
            }
            sample.f = evaluate(wo, sample.wi);
            return sample;
        }
    };
    class SpecularReflection : public BSDFClosure {
        Spectrum R;

      public:
        SpecularReflection(const Spectrum &R) : R(R) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.0f; }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const { return Spectrum(0.0f); }
        [[nodiscard]] BSDFType type() const { return BSDFType::SpecularReflection; }
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            sample.wi = glm::reflect(-wo, vec3(0, 1, 0));
            sample.sampled = type();
            sample.pdf = 1.0;
            sample.f = R / (std::abs(cos_theta(sample.wi)));
            return sample;
        }
    };
    class SpecularTransmission : public BSDFClosure {
        Spectrum R;
        Float eta;

      public:
        SpecularTransmission(const Spectrum &R, Float eta) : R(R), eta(eta) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.0f; }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const { return Spectrum(0.0f); }
        [[nodiscard]] BSDFType type() const { return BSDFType::SpecularTransmission; }
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            Float etaIO = same_hemisphere(wo, vec3(0, 1, 0)) ? eta : 1.0f / eta;
            auto wt = refract(wo, faceforward(wo, vec3(0, 1, 0)), etaIO);
            if (!wt) {
                return std::nullopt;
            }
            sample.wi = *wt;
            sample.sampled = type();
            sample.pdf = 1.0;
            sample.f = R / (std::abs(cos_theta(sample.wi)));
            return sample;
        }
    };

    class AKR_EXPORT FresnelSpecular : public BSDFClosure {
        const Spectrum R, T;
        const Float etaA, etaB;
        const FresnelDielectric fresnel;

      public:
        explicit FresnelSpecular(const Spectrum &R, const Spectrum &T, Float etaA, Float etaB)
            : R(R), T(T), etaA(etaA), etaB(etaB), fresnel(etaA, etaB) {}
        [[nodiscard]] BSDFType type() const { return BSDFType::SpecularTransmission | BSDFType::SpecularReflection; }
        [[nodiscard]] Float evaluate_pdf(const vec3 &wo, const vec3 &wi) const override { return 0; }
        [[nodiscard]] Spectrum evaluate(const vec3 &wo, const vec3 &wi) const override { return Spectrum(0); }
        [[nodiscard]] std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const override;
    };
    class PassThrough : public BSDFClosure {
        Spectrum R;

      public:
        PassThrough(const Spectrum &R) : R(R) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.0f; }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const { return Spectrum(0.0f); }
        [[nodiscard]] BSDFType type() const { return BSDFType::SpecularTransmission; }
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            sample.wi = -wo;
            sample.sampled = type();
            sample.pdf = 1.0;
            sample.f = R / (std::abs(cos_theta(sample.wi)));
            return sample;
        }
    };
    class AKR_EXPORT MixBSDF : public BSDFClosure {
      public:
        Float fraction;
        const BSDFClosure *bsdf_A = nullptr;
        const BSDFClosure *bsdf_B = nullptr;
        MixBSDF() = default;
        MixBSDF(Float fraction, const BSDFClosure *bsdf_A, const BSDFClosure *bsdf_B)
            : fraction(fraction), bsdf_A(bsdf_A), bsdf_B(bsdf_B) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const override {
            return (1.0 - fraction) * bsdf_A->evaluate_pdf(wo, wi) + fraction * bsdf_B->evaluate_pdf(wo, wi);
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const override {
            return (1.0 - fraction) * bsdf_A->evaluate(wo, wi) + fraction * bsdf_B->evaluate(wo, wi);
        }
        [[nodiscard]] BSDFType type() const override { return BSDFType(bsdf_A->type() | bsdf_B->type()); }
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const override {
            BSDFSample sample;
            std::optional<BSDFSample> inner_sample;
            Float pdf_select = 0;
            if (u[0] < fraction) {
                vec2 u_(u[0] / fraction, u[1]);
                pdf_select = fraction;
                inner_sample = bsdf_B->sample(u_, wo);
            } else {
                vec2 u_((u[0] - fraction) / (1.0 - fraction), u[1]);
                pdf_select = 1.0 - fraction;
                inner_sample = bsdf_A->sample(u_, wo);
            }
            sample = *inner_sample;
            sample.pdf *= pdf_select;
            return sample;
        }
    };
} // namespace akari::render