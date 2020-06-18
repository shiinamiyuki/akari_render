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
#include <akari/render/bsdf.h>

namespace akari {
    Float BSDF::evaluate_pdf(const vec3 &woW, const vec3 &wiW) const {
        auto wo = world_to_local(woW);
        auto wi = world_to_local(wiW);
        std::array<Float, MaxBSDF> importance;
        evaluate_importance(wo, importance);
        Float pdf = 0;
        int cnt = 0;
        for (int i = 0; i < nComp; i++) {
            auto *comp = components[i];
            if (!comp->is_delta()) {

                pdf += comp->evaluate_pdf(wo, wi) * importance[i];
            }
            cnt++;
        }
        return pdf;
    }
    Spectrum BSDF::evaluate(const vec3 &woW, const vec3 &wiW) const {
        auto wo = world_to_local(woW);
        auto wi = world_to_local(wiW);
        Spectrum f(0);
        for (int i = 0; i < nComp; i++) {
            auto *comp = components[i];
            if (!comp->is_delta()) {
                auto reflect = (dot(woW, Ns) * dot(wiW, Ns) > 0);
                if ((reflect && comp->match_flags(BSDF_REFLECTION)) ||
                    (!reflect && comp->match_flags(BSDF_TRANSMISSION))) {
                    f += comp->evaluate(wo, wi);
                }
            }
        }
        return f;
    }
    void BSDF::evaluate_importance(const vec3 &wo, std::array<Float, MaxBSDF> &func) const {
        Float funcInt = 0.0f;
        for (size_t i = 0; i < nComp; i++) {
            func[i] = components[i]->importance(wo);
            funcInt += func[i];
        }
        for (size_t i = 0; i < nComp; i++) {
            func[i] /= funcInt;
        }
    }
    void BSDF::sample(BSDFSample &sample) const {
        if (nComp == 0) {
            return;
        }
        std::array<Float, MaxBSDF> importance;
        vec3 wo, wi;
        wo = world_to_local(sample.wo);
        evaluate_importance(wo, importance);
        // int selected = std::clamp(int(sample.u0 * (float)nComp), 0, nComp - 1);
        int selected = 0;
        {
            Float _int = 0.0f;
            for (size_t i = 0; i < nComp; i++) {
                _int += importance[i];
                if (sample.u0 < _int) {
                    sample.u0 = clamp01((sample.u0 - _int) / importance[i]);
                    break;
                }
                selected++;
            }
            selected = std::min(selected, nComp - 1);
        }

        {
            auto *comp = components[selected];
            sample.f = comp->sample(sample.u, wo, &wi, &sample.pdf, &sample.sampledType);
            if (sample.pdf <= 0) {
                return;
            }
            sample.wi = local_to_world(wi);
            sample.pdf *= importance[selected];
            if (sample.sampledType & BSDF_SPECULAR) {
                return;
            }
        }
        auto &f = sample.f;
        auto woW = local_to_world(wo);
        auto wiW = local_to_world(wi);
        for (int i = 0; i < nComp; i++) {
            if (i == selected)
                continue;
            auto *comp = components[i];

            auto reflect = (dot(woW, Ns) * dot(wiW, Ns) > 0);
            if ((reflect && comp->match_flags(BSDF_REFLECTION)) || (!reflect && comp->match_flags(BSDF_TRANSMISSION))) {
                f += comp->evaluate(wo, wi);
            }
            sample.pdf += comp->evaluate_pdf(wo, wi) * importance[i];
        }
    }
} // namespace akari
