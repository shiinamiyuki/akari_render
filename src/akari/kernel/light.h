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
#include <akari/kernel/texture.h>
#include <akari/kernel/sampling.h>
namespace akari {
    AKR_VARIANT struct LightSample {
        AKR_IMPORT_TYPES()
        Normal3f ng = Normal3f(0.0f);
        Vector3f wi; // w.r.t to the luminated surface; normalized
        Float pdf = 0.0f;
        Spectrum L;
        Ray3f shadow_ray;
    };
    AKR_VARIANT struct LightSampleContext {
        AKR_IMPORT_TYPES()
        Point2f u;
        Point3f p;
    };
    AKR_VARIANT
    struct DirectLighting {
        AKR_IMPORT_TYPES()
        Ray3f shadow_ray;
        Spectrum color;
        Float pdf;
    };
    // Diffuse Area Light
    AKR_VARIANT class AreaLight {
      public:
        AKR_IMPORT_TYPES()
        const Texture<C> *color;
        bool double_sided = false;
        Triangle<C> triangle;
        AreaLight() = default;
        AKR_XPU AreaLight(Triangle<C> triangle) : triangle(triangle) {
            color = triangle.material->template get<EmissiveMaterial<C>>()->color;
            double_sided = triangle.material->template get<EmissiveMaterial<C>>()->double_sided;
        }
        AKR_XPU LightSample<C> sample(const LightSampleContext<C> &ctx) const {
            auto coords = sampling<C>::uniform_sample_triangle(ctx.u);
            auto p = triangle.p(coords);
            LightSample<C> sample;
            sample.ng = triangle.ng();
            sample.wi = p - ctx.p;
            auto dist_sqr = dot(sample.wi, sample.wi);
            sample.wi /= sqrt(dist_sqr);
            sample.L = color->evaluate(triangle.texcoord(coords));
            sample.pdf = dist_sqr / (std::abs(dot(sample.wi, sample.ng))) / triangle.area();
            sample.shadow_ray = Ray3f(p, -sample.wi, Constants<Float>::Eps() / std::abs(dot(sample.wi, sample.ng)),
                                      sqrt(dist_sqr) * (Float(1.0f) - Constants<Float>::ShadowEps()));
            return sample;
        }
    };
    template <class C>
    using Light = AreaLight<C>;
} // namespace akari