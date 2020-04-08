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
#include <Akari/Render/BSDF.h>

namespace Akari {
    Float BSDF::EvaluatePdf(const vec3 &woW, const vec3 &wiW) const {
        auto wo = WorldToLocal(woW);
        auto wi = WorldToLocal(wiW);
        Float pdf = 0;
        int cnt = 0;
        for (int i = 0; i < nComp; i++) {
            auto *comp = components[i];
            if (!comp->IsDelta()) {

                pdf += comp->EvaluatePdf(wo, wi);
            }
            cnt++;
        }
        if (cnt > 1) {
            pdf /= (float)cnt;
        }
        return pdf;
    }
    Spectrum BSDF::Evaluate(const vec3 &woW, const vec3 &wiW) const {
        auto wo = WorldToLocal(woW);
        auto wi = WorldToLocal(wiW);
        Spectrum f(0);
        for (int i = 0; i < nComp; i++) {
            auto *comp = components[i];
            if (!comp->IsDelta()) {
                auto reflect = (dot(woW, Ns) * dot(wiW, Ns) > 0);
                if ((reflect && comp->MatchFlag(BSDF_REFLECTION)) || (!reflect && comp->MatchFlag(BSDF_TRANSMISSION))) {
                    f += comp->Evaluate(wo, wi);
                }
            }
        }
        return f;
    }
    void BSDF::Sample(BSDFSample &sample) const {
        if (nComp == 0) {
            return;
        }
        int selected = std::clamp(int(sample.u0 * (float)nComp), 0, nComp - 1);
        sample.u0 = std::min(sample.u0 * (float)nComp - (float)selected, 1.0f - 1e-7f);
        vec3 wo, wi;
        wo = WorldToLocal(sample.wo);
        {
            auto *comp = components[selected];
            sample.f = comp->Sample(sample.u, wo, &wi, &sample.pdf, &sample.sampledType);
            sample.wi = LocalToWorld(wi);
            if (comp->IsDelta()) {
                return;
            }
        }
        auto &f = sample.f;
        auto woW = LocalToWorld(wo);
        auto wiW = LocalToWorld(wi);
        for (int i = 0; i < nComp; i++) {
            if (i == selected)
                continue;
            auto *comp = components[i];

            auto reflect = (dot(woW, Ns) * dot(wiW, Ns) > 0);
            if ((reflect && comp->MatchFlag(BSDF_REFLECTION)) || (!reflect && comp->MatchFlag(BSDF_TRANSMISSION))) {
                f += comp->Evaluate(wo, wi);
            }
            sample.pdf += comp->EvaluatePdf(wo, wi);
        }
        if (nComp > 1) {
            sample.pdf /= nComp;
        }
    }
} // namespace Akari
