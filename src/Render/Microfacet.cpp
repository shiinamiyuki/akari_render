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
#include <Akari/Render/Microfacet.h>
namespace Akari {

    Float MicrofacetReflection::EvaluatePdf(const vec3 &wo, const vec3 &wi) const {
        if (!SameHemisphere(wo, wi))
            return 0.0f;
        auto wh = normalize(wo + wi);
        return microfacet.evaluatePdf(wh) / (4.0f * dot(wo, wh));
    }
    Spectrum MicrofacetReflection::Evaluate(const vec3 &wo, const vec3 &wi) const {
        if (!SameHemisphere(wo, wi))
            return {};
        Float cosThetaO = AbsCosTheta(wo);
        Float cosThetaI = AbsCosTheta(wi);
        auto wh = (wo + wi);
        if (cosThetaI == 0 || cosThetaO == 0)
            return Spectrum(0);
        if (wh.x == 0 && wh.y == 0 && wh.z == 0)
            return Spectrum(0);
        wh = normalize(wh);
        auto F = fresnel->Evaluate(dot(wi, wh));
        return R * F * (microfacet.D(wh) * microfacet.G(wo, wi, wh) * F / (4.0f * cosThetaI * cosThetaO));
    }
    Spectrum MicrofacetReflection::Sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf,
                                          BSDFType *sampledType) const {
        *sampledType = type;
        auto wh = microfacet.sampleWh(wo, u);
        *wi = Reflect(wo, wh);
        if (!SameHemisphere(wo, *wi)) {
            *pdf = 0;
            return Spectrum(0);
        } else {
            *pdf = microfacet.evaluatePdf(wh) / (4.0f * dot(wo, wh));
        }
        return Evaluate(wo, *wi);
    }

    Spectrum MicrofacetTransmission::Evaluate(const vec3 &wo, const vec3 &wi) const {
        if (SameHemisphere(wo, wi))
            return {};
        Float cosThetaO = AbsCosTheta(wo);
        Float cosThetaI = AbsCosTheta(wi);
        if (cosThetaI == 0 || cosThetaO == 0)
            return Spectrum(0);
        Float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
        vec3 wh = normalize(wo + wi * eta);
        if (wh.y < 0)
            wh = -wh;
        auto F = fresnel.Evaluate(dot(wo, wh));
        auto D = microfacet.D(wh);
        auto G = microfacet.G(wo, wi, wh);
        auto sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        auto denom = sqrtDenom * sqrtDenom;
        auto factor = abs(dot(wi, wh) * dot(wo, wh)) / (cosThetaI * cosThetaO);
        return (Spectrum(1) - F) * T * D * G / denom * factor;
    }
    Float MicrofacetTransmission::EvaluatePdf(const vec3 &wo, const vec3 &wi) const {
        if (!SameHemisphere(wo, wi))
            return 0.0f;
        Float eta = CosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
        vec3 wh = normalize(wo + wi * eta);
        Float sqrtDenom = dot(wo, wh) + eta * dot(wi, wh);
        Float dwh_dwi = std::abs((eta * eta * dot(wi, wh)) / (sqrtDenom * sqrtDenom));
        return microfacet.evaluatePdf(wh) * dwh_dwi;
    }
    Spectrum MicrofacetTransmission::Sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf,
                                            BSDFType *sampledType) const {
        *sampledType = type;
        if (wo.y == 0) {
            return Spectrum(0);
        }
        auto wh = microfacet.sampleWh(wo, u);
        if (dot(wo, wh) < 0)
            return Spectrum(0);
        Float eta = CosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
        if (!Refract(wo, wh, eta, wi))
            return Spectrum(0);
        *pdf = EvaluatePdf(wo, *wi);
        return Evaluate(wo, *wi);
    }

} // namespace Akari
