

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
#include <akari/core/math.h>
#include <akari/render/scenegraph.h>
#include <akari/shaders/common.h>
#include <akari/core/memory.h>
#include <akari/render/shape.h>
#include <akari/render/texture.h>
#include <optional>

namespace akari::render {
    struct LightSample {
        Vec3 ng = Vec3(0.0f);
        Vec3 wi; // w.r.t to the luminated surface; normalized
        Float pdf = 0.0f;
        Spectrum L;
        Ray shadow_ray;
    };
    struct LightSampleContext {

        vec2 u;
        Vec3 p;
    };

    struct DirectLighting {
        Ray shadow_ray;
        Spectrum color;
        Float pdf;
    };
    class Light {
      public:
        virtual LightSample sample(const LightSampleContext &ctx) const = 0;
    };
    class AreaLight final : public Light {
      public:
        const Texture *color;
        bool double_sided = false;
        Triangle triangle;
        AreaLight() = default;
        AreaLight(Triangle triangle) : triangle(triangle) {
            color = triangle.material->as_emissive()->color;
            double_sided = triangle.material->as_emissive()->double_sided;
        }
        LightSample sample(const LightSampleContext &ctx) const override {
            auto coords = shader::uniform_sample_triangle(ctx.u);
            auto p = triangle.p(coords);
            LightSample sample;
            sample.ng = triangle.ng();
            sample.wi = p - ctx.p;
            auto dist_sqr = dot(sample.wi, sample.wi);
            sample.wi /= sqrt(dist_sqr);
            sample.L = color->evaluate(triangle.texcoord(coords));
            sample.pdf = dist_sqr / max(Float(0.0), -dot(sample.wi, sample.ng)) / triangle.area();
            sample.shadow_ray = Ray(p, -sample.wi, Eps / std::abs(dot(sample.wi, sample.ng)),
                                      sqrt(dist_sqr) * (Float(1.0f) - ShadowEps));
            return sample;
        }
    };
} // namespace akari::render