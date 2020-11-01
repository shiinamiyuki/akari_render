

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

#include <akari/render/closure.h>

namespace akari::render {
    Spectrum FresnelNoOp::evaluate(Float cosThetaI) const { return Spectrum(1.0f); }
    Spectrum FresnelConductor::evaluate(Float cosThetaI) const { return fr_conductor(cosThetaI, etaI, etaT, k); }
    Spectrum FresnelDielectric::evaluate(Float cosThetaI) const {
        return Spectrum(fr_dielectric(cosThetaI, etaI, etaT));
    }
    [[nodiscard]] std::optional<BSDFSample> FresnelSpecular::sample(const vec2 &u, const Vec3 &wo) const {
        Float F = fr_dielectric(cos_theta(wo), etaA, etaB);
        AKR_ASSERT(F >= 0.0);
        BSDFSample sample;
        if (u[0] < F) {
            sample.wi = reflect(-wo, vec3(0, 1, 0));
            sample.pdf = F;
            sample.sampled = BSDFType::SpecularReflection;
            sample.f = F * R / abs_cos_theta(sample.wi);
        } else {
            bool entering = cos_theta(wo) > 0;
            Float etaI = entering ? etaA : etaB;
            Float etaT = entering ? etaB : etaA;
            auto wt = refract(wo, faceforward(wo, vec3(0, 1, 0)), etaI / etaT);
            if (!wt) {
                AKR_ASSERT(etaI > etaT);
                return std::nullopt;
            }
            Spectrum ft = T * (1 - F);
            sample.sampled = BSDFType::SpecularTransmission;

            ft *= (etaI * etaI) / (etaT * etaT);
            sample.pdf = 1 - F;
            sample.wi = *wt;
            sample.f = ft / abs_cos_theta(sample.wi);
        }
        return sample;
    }
} // namespace akari::render