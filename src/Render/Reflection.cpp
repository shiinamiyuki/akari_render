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

#include <Akari/Render/Reflection.h>

namespace Akari {
    Spectrum FresnelNoOp::Evaluate(Float cosThetaI) const { return Spectrum(1.0f); }
    Spectrum FresnelConductor::Evaluate(Float cosThetaI) const { return FrConductor(cosThetaI, etaI, etaT, k); }
    Spectrum FresnelDielectric::Evaluate(Float cosThetaI) const {
        return Spectrum(FrDielectric(cosThetaI, etaI, etaT));
    }
    Spectrum LambertianReflection::Evaluate(const vec3 &wo, const vec3 &wi) const {
        if (!SameHemisphere(wo, wi)) {
            return Spectrum(0);
        }
        return R * InvPi;
    }

    Spectrum SpecularReflection::Sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf,
                                        BSDFType *sampledType) const {
        *wi = Reflect(wo, vec3(0, 1, 0));
        *pdf = 1;
        *sampledType = type;
        return fresnel->Evaluate(CosTheta(*wi)) * R / AbsCosTheta(*wi);
    }

    Spectrum SpecularTransmission::Sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf,
                                          BSDFType *sampledType) const {
        bool entering = CosTheta(wo) > 0;
        Float etaI = entering ? etaA : etaB;
        Float etaT = entering ? etaB : etaA;

        if (!Refract(wo, FaceForward(vec3(0, 0, 1), wo), etaI / etaT, wi))
            return Spectrum(0);
        *sampledType = type;
        *pdf = 1;
        Spectrum ft = T * (Spectrum(1) - fresnel.Evaluate(CosTheta(*wi)));
        if (mode == TransportMode::ERadiance)
            ft *= (etaI * etaI) / (etaT * etaT);
        return ft / AbsCosTheta(*wi);
    }
    Spectrum FresnelSpecular::Sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf, BSDFType *sampledType) const {
        Float F = FrDielectric(CosTheta(wo), etaA, etaB);
        if (u[0] < F) {
            *wi = Reflect(wo, vec3(0, 1, 0));
            *pdf = 1;
            *sampledType = BSDFType(BSDF_SPECULAR | BSDF_REFLECTION);
            return fresnel.Evaluate(CosTheta(*wi)) * R / AbsCosTheta(*wi);
        } else {
            bool entering = CosTheta(wo) > 0;
            Float etaI = entering ? etaA : etaB;
            Float etaT = entering ? etaB : etaA;
            if (!Refract(wo, FaceForward(vec3(0, 0, 1), wo), etaI / etaT, wi))
                return Spectrum(0);
            Spectrum ft = T * (1 - F);
            if (sampledType)
                *sampledType = BSDFType(BSDF_SPECULAR | BSDF_TRANSMISSION);
            *pdf = 1 - F;
            return ft / AbsCosTheta(*wi);
        }
    }
} // namespace Akari
