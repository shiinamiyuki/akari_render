

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
#include <akari/render/interaction.h>

namespace akari::render {
    enum BSDFType : int {
        BSDF_NONE = 0u,
        BSDF_REFLECTION = 1u << 0u,
        BSDF_TRANSMISSION = 1u << 1u,
        BSDF_DIFFUSE = 1u << 2u,
        BSDF_GLOSSY = 1u << 3u,
        BSDF_SPECULAR = 1u << 4u,
        BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION,
    };
    class BSDFClosure {
      public:
        [[nodiscard]] virtual Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const = 0;
        [[nodiscard]] virtual Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const = 0;
        [[nodiscard]] virtual BSDFType type() const = 0;
        [[nodiscard]] bool match_flags(BSDFType flag) const { return ((uint32_t)type() & (uint32_t)flag) != 0; }
        virtual Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const = 0;
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
        [[nodiscard]] BSDFType type() const override { return BSDFType(BSDF_DIFFUSE | BSDF_REFLECTION); }
        Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const override {

            *wi = cosine_hemisphere_sampling(u);
            if (!same_hemisphere(wo, *wi)) {
                wi->y = -wi->y;
            }
            *sampledType = type();
            *pdf = cosine_hemisphere_pdf(std::abs(cos_theta(*wi)));
            return R * InvPi;
        }
    };

    class MicrofacetReflection : public BSDFClosure {

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
        [[nodiscard]] BSDFType type() const override { return BSDFType(BSDF_GLOSSY | BSDF_REFLECTION); }
        Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const override {
            *sampledType = type();
            auto wh = microfacet_sample_wh(model, wo, u);
            *wi = glm::reflect(-wo, wh);
            if (!same_hemisphere(wo, *wi)) {
                *pdf = 0;
                return Spectrum(0);
            } else {
                if (wh.y < 0) {
                    wh = -wh;
                }
                *pdf = microfacet_evaluate_pdf(model, wh) / (Float(4.0f) * abs(dot(wo, wh)));
            }
            return evaluate(wo, *wi);
        }
    };

    class MixBSDF : public BSDFClosure {
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
        Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const override {
            Float pdf_inner = 0;
            Float pdf_select = 0;
            Spectrum f;
            if (u[0] < fraction) {
                vec2 u_(u[0] / fraction, u[1]);
                pdf_select = fraction;
                f = bsdf_B->sample(u_, wo, wi, &pdf_inner, sampledType);
            } else {
                vec2 u_((u[0] - fraction) / (1.0 - fraction), u[1]);
                pdf_select = 1.0 - fraction;
                f = bsdf_A->sample(u_, wo, wi, &pdf_inner, sampledType);
            }
            *pdf = pdf_inner * pdf_select;
            return f;
        }
    };

} // namespace akari::render