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

#include <Akari/Core/Application.h>
#include <Akari/Core/Logger.h>
#include <Akari/Render/Microfacet.h>
#include <Akari/Render/Sampler.h>
using namespace Akari;
bool RunTestBSDF_SamplePdf(const vec3 &wo, const BSDFComponent *bsdf, Sampler *sampler, size_t N = 100000u) {
    double sum = 0;
    size_t good = 0, bad = 0;
    for (size_t i = 0; i < N; i++) {
        vec3 wi;
        BSDFType sampledType;
        Float pdf = 0;
        auto f = bsdf->Sample(sampler->Next2D(), wo, &wi, &pdf, &sampledType);
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
    Info("good: {}, bad: {}\n", good, bad);
    Info("integral = {}, abs_diff = {}, rel_diff = {}\n", actual, abs(expected - actual),
         abs((expected - actual) / actual));
    return true;
}

int main() {
    Application app;
    try {
        vec3 wo = normalize(vec3(0.3, 1, 0));
        Info("Test BSDFComponent::Sample()\n");
        {
            auto sampler = Cast<Sampler>(CreateComponent("RandomSampler"));
            LambertianReflection bsdf(Spectrum(1));
            RunTestBSDF_SamplePdf(wo, &bsdf, sampler.get());
        }
        {
            auto sampler = Cast<Sampler>(CreateComponent("RandomSampler"));
            FresnelDielectric fresnel(1.0, 1.3);
            MicrofacetReflection bsdf(Spectrum(1), MicrofacetModel(MicrofacetType::EGGX, 0.1), &fresnel);
            RunTestBSDF_SamplePdf(wo, &bsdf, sampler.get());
        }
    } catch (std::exception &e) {
        Fatal("Exception: {}", e.what());
        exit(1);
    }
}