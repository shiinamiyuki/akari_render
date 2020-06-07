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

#include <akari/core/application.h>
#include <akari/core/logger.h>
#include <akari/render/microfacet.h>
#include <akari/render/sampler.h>
using namespace akari;
bool RunTestBSDF_SamplePdf(const vec3 &wo, const BSDFComponent *bsdf, Sampler *sampler, size_t N = 10000000u) {
    double sum = 0;
    size_t good = 0, bad = 0;
    for (size_t i = 0; i < N; i++) {
        vec3 wi;
        BSDFType sampledType;
        Float pdf = 0;
        auto f = bsdf->sample(sampler->next2d(), wo, &wi, &pdf, &sampledType);
        (void)f;
        if (std::isnan(pdf) || pdf <= 0) {
            bad++;
        } else {
            sum += 1.0 / pdf;
            good++;
        }
    }
    double expected = Pi * 2;
    auto actual = sum / good;
    info("good: {}, bad: {}\n", good, bad);
    info("integral = {}, abs_diff = {}, rel_diff = {}\n", actual, abs(expected - actual),
         abs((expected - actual) / actual));
    return true;
}
bool RunTestMicrofacet(const vec3 &wo, const MicrofacetModel* model, Sampler *sampler, size_t N = 10000000u) {
    double sum = 0;
    size_t good = 0, bad = 0;
    for (size_t i = 0; i < N; i++) {
        vec3 wi;
        (void)wi;
        BSDFType sampledType;
        (void)sampledType;
        Float pdf = 0;
        auto wh = model->sample_wh(wo, sampler->next2d());(void)wh;
        pdf = model->evaluate_pdf(wh);
        if (std::isnan(pdf) || pdf <= 0) {
            bad++;
        } else {
            sum += 1.0 / pdf;
            good++;
        }
    }
    double expected = Pi * 2;
    auto actual = sum / good;
    info("good: {}, bad: {}\n", good, bad);
    info("integral = {}, abs_diff = {}, rel_diff = {}\n", actual, abs(expected - actual),
         abs((expected - actual) / actual));
    return true;
}
int main() {
    Application app;
    try {
        vec3 wo = normalize(vec3(0, 1, 0));
        info("Test BSDFComponent::Sample()\n");
        {
            auto sampler = dyn_cast<Sampler>(create_component("RandomSampler"));
            LambertianReflection bsdf(Spectrum(1));
            RunTestBSDF_SamplePdf(wo, &bsdf, sampler.get());
        }
        {
            auto sampler = dyn_cast<Sampler>(create_component("RandomSampler"));
            FresnelDielectric fresnel(1.0, 1.3);
            MicrofacetModel model(MicrofacetType::EGGX, 0.3);
            RunTestMicrofacet(wo,&model, sampler.get());
            MicrofacetReflection bsdf(Spectrum(1), model, &fresnel);
            RunTestBSDF_SamplePdf(wo, &bsdf, sampler.get());
        }
    } catch (std::exception &e) {
        fatal("Exception: {}", e.what());
        exit(1);
    }
}