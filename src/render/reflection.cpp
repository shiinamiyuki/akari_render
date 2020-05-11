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

#include <akari/core/logger.h>
#include <akari/render/reflection.h>

namespace akari {
    Spectrum FresnelNoOp::evaluate(Float cosThetaI) const { return Spectrum(1.0f); }
    Spectrum FresnelConductor::evaluate(Float cosThetaI) const { return FrConductor(cosThetaI, etaI, etaT, k); }
    Spectrum FresnelDielectric::evaluate(Float cosThetaI) const {
        return Spectrum(fr_dielectric(cosThetaI, etaI, etaT));
    }
    Spectrum LambertianReflection::evaluate(const vec3 &wo, const vec3 &wi) const {
        if (!same_hemisphere(wo, wi)) {
            return Spectrum(0);
        }
        return R * InvPi;
    }

    Spectrum SpecularReflection::sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf,
                                        BSDFType *sampledType) const {
        *wi = reflect(wo, vec3(0, 1, 0));
        *pdf = 1;
        *sampledType = type;
        return fresnel->evaluate(cos_theta(*wi)) * R / abs_cos_theta(*wi);
    }

    Spectrum SpecularTransmission::sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf,
                                          BSDFType *sampledType) const {
        bool entering = cos_theta(wo) > 0;
        Float etaI = entering ? etaA : etaB;
        Float etaT = entering ? etaB : etaA;

        if (!refract(wo, face_forward(vec3(0, 1, 0), wo), etaI / etaT, wi))
            return Spectrum(0);
        *wi = -wo;
        *sampledType = type;
        *pdf = 1;
        Spectrum ft = T * (Spectrum(1) - fresnel.evaluate(cos_theta(*wi)));
        if (mode == TransportMode::ERadiance)
            ft *= (etaI * etaI) / (etaT * etaT);
        return ft / abs_cos_theta(*wi);
    }
    Spectrum FresnelSpecular::sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf, BSDFType *sampledType) const {
        Float F = fr_dielectric(cos_theta(wo), etaA, etaB);
        AKARI_ASSERT(F >= 0 && F <= 1);
        if (u[0] < F) {
            *wi = reflect(wo, vec3(0, 1, 0));
            *pdf = F;
            *sampledType = BSDFType(BSDF_SPECULAR | BSDF_REFLECTION);
            return F * R / abs_cos_theta(*wi);
        } else {
            bool entering = cos_theta(wo) > 0;
            Float etaI = entering ? etaA : etaB;
            Float etaT = entering ? etaB : etaA;
            //            Debug("{}\n", etaI / etaT);
            if (!refract(wo, face_forward(vec3(0, 1, 0), wo), etaI / etaT, wi))
                return Spectrum(0);
            Spectrum ft = T * (1 - F);
            *sampledType = BSDFType(BSDF_SPECULAR | BSDF_TRANSMISSION);
            if (mode == TransportMode::ERadiance)
                ft *= (etaI * etaI) / (etaT * etaT);
            *pdf = 1 - F;
            //            Info("{} {} {} {}\n", ft[0],ft[1],ft[2],  AbsCosTheta(*wi));
            return ft / abs_cos_theta(*wi);
        }
    }

    Spectrum OrenNayar::evaluate(const vec3 &wo, const vec3 &wi) const {
        Float sinThetaI = sin_theta(wi);
        Float sinThetaO = sin_theta(wo);
        Float maxCos = 0;
        if (sinThetaI > 1e-4f && sinThetaO > 1e-4f) {
            Float sinPhiI = sin_phi(wi), cosPhiI = cos_phi(wi);
            Float sinPhiO = sin_phi(wo), cosPhiO = cos_phi(wo);
            Float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
            maxCos = std::max((Float)0, dCos);
        }
        Float sinAlpha, tanBeta;
        if (abs_cos_theta(wi) > abs_cos_theta(wo)) {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / abs_cos_theta(wi);
        } else {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / abs_cos_theta(wo);
        }
        return R * InvPi * (A + B * maxCos * sinAlpha * tanBeta);
    }
} // namespace akari
