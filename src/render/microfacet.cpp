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
#include <akari/render/microfacet.h>
namespace akari {

    Float MicrofacetReflection::evaluate_pdf(const vec3 &wo, const vec3 &wi) const {
        if (!same_hemisphere(wo, wi))
            return 0.0f;
        auto wh = normalize(wo + wi);
        return microfacet.evaluate_pdf(wh) / (4.0f * dot(wo, wh));
    }
    Float MicrofacetReflection::importance(const vec3 &wo) const{
        return R.luminance() * fresnel->evaluate(abs_cos_theta(wo)).luminance()* 0.9f + 0.1f;
    }
    Spectrum MicrofacetReflection::evaluate(const vec3 &wo, const vec3 &wi) const {
        if (!same_hemisphere(wo, wi))
            return {};
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
        auto F = fresnel->evaluate(dot(wi, wh));
        return R * (microfacet.D(wh) * microfacet.G(wo, wi, wh) * F / (4.0f * cosThetaI * cosThetaO));
    }
    Spectrum MicrofacetReflection::sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf,
                                          BSDFType *sampledType) const {
        *sampledType = type;
        auto wh = microfacet.sample_wh(wo, u);
        *wi = reflect(wo, wh);
        if (!same_hemisphere(wo, *wi)) {
            *pdf = 0;
            return Spectrum(0);
        } else {
            if (wh.y < 0) {
                wh = -wh;
            }
            *pdf = microfacet.evaluate_pdf(wh) / (4.0f * abs(dot(wo, wh)));
        }
        return evaluate(wo, *wi);
    }

    Spectrum MicrofacetTransmission::evaluate(const vec3 &wo, const vec3 &wi) const {
        if (same_hemisphere(wo, wi))
            return {};
        Float cosThetaO = abs_cos_theta(wo);
        Float cosThetaI = abs_cos_theta(wi);
        if (cosThetaI == 0 || cosThetaO == 0)
            return Spectrum(0);
        Float eta = cos_theta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
        vec3 wh = normalize(wo + wi * eta);
        if (wh.y < 0)
            wh = -wh;
        auto F = fresnel.evaluate(dot(wo, wh));
        auto D = microfacet.D(wh);
        auto G = microfacet.G(wo, wi, wh);
        auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        auto denom = sqrtDenom * sqrtDenom;
        Float k = (mode == TransportMode::ERadiance) ? (1.0f / eta) : 1.0f;
        auto factor = abs(dot(wi, wh) * dot(wo, wh) * k * k) / (cosThetaI * cosThetaO);
        return (Spectrum(1) - F) * T * std::abs(D * G / denom * factor);
    }
    Float MicrofacetTransmission::importance(const vec3 &wo) const{
        return T.luminance() *  fr_dielectric(cos_theta(wo), etaA, etaB) * 0.9f + 0.1f;
    }
    Float MicrofacetTransmission::evaluate_pdf(const vec3 &wo, const vec3 &wi) const {
        if (same_hemisphere(wo, wi))
            return 0.0f;
        Float eta = cos_theta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
        vec3 wh = normalize(wo + wi * eta);
        if (wh.y < 0)
            wh = -wh;
        Float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        Float dwh_dwi = std::abs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
        return microfacet.evaluate_pdf(wh) * dwh_dwi;
    }
    Spectrum MicrofacetTransmission::sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf,
                                            BSDFType *sampledType) const {
        *sampledType = type;
        if (wo.y == 0) {
            return Spectrum(0);
        }
        auto wh = microfacet.sample_wh(wo, u);
        if (dot(wo, wh) < 0)
            return Spectrum(0);
        Float eta = cos_theta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
        if (!refract(wo, wh, eta, wi))
            return Spectrum(0);
        *pdf = evaluate_pdf(wo, *wi);
        return evaluate(wo, *wi);
    }
    Spectrum FresnelGlossy::sample(const vec2 &u0, const vec3 &wo, vec3 *wi, Float *pdf, BSDFType *sampledType) const {
        Float F = fr_dielectric(cos_theta(wo), etaA, etaB);
        vec2 u = u0;
        AKR_ASSERT(F >= 0 && F <= 1);
        if (u[0] < F) {
            u[0] /= F;
            auto f = brdf.sample(u, wo, wi, pdf, sampledType);
            *pdf *= F;
            //  *wi = reflect(wo, vec3(0, 1, 0));
            //     *pdf = F;
            //     *sampledType = BSDFType(BSDF_SPECULAR | BSDF_REFLECTION);
            //     return F * R / abs_cos_theta(*wi);
            return f;
        } else {
            u[0] = (u[0] - F) / (1.0f - F);
            auto f = btdf.sample(u, wo, wi, pdf, sampledType);
            *pdf *= (1.0f - F);
            return f;
        }
    }
    Float FresnelGlossy::evaluate_pdf(const vec3 &wo, const vec3 &wi) const {
        Float F = fr_dielectric(cos_theta(wo), etaA, etaB);
        return F * brdf.evaluate_pdf(wo, wi) + (1.0f - F) * btdf.evaluate_pdf(wo, wi);
    }
    Spectrum FresnelGlossy::evaluate(const vec3 &wo, const vec3 &wi) const {
        return brdf.evaluate(wo, wi) + btdf.evaluate(wo, wi);
    }
} // namespace akari
