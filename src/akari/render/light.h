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
#include <akari/render/common.h>
#include <akari/core/memory.h>
#include <akari/render/shape.h>
#include <akari/render/texture.h>
#include <akari/render/material.h>
#include <akari/render/endpoint.h>
#include <optional>

namespace akari::render {

    struct LightSampleContext {
        vec2 u;
        Vec3 p;
    };
    struct LightSample {
        Vec3 ng = Vec3(0.0f);
        Vec3 wi; // w.r.t to the luminated surface; normalized
        Float pdf = 0.0f;
        Spectrum I;
        Ray shadow_ray;
    };
    struct LightRaySample {
        Ray ray;
        Spectrum E;
        vec3 ng;
        vec2 uv; // 2D parameterized position
        Float pdfPos = 0.0, pdfDir = 0.0;
    };
    struct DirectLighting {
        Ray shadow_ray;
        Spectrum color;
        Float pdf;
    };

    class Light : public EndPoint<LightSample, LightRaySample, LightSampleContext> {
      public:
        virtual Spectrum Le(const Vec3 &wo, const ShadingPoint &sp) const = 0;
        virtual Float power() const = 0;
    };
    class Scene;
    class LightNode : public SceneGraphNode {
      public:
        // triangle is given if the light is attached to a triangle
        virtual std::shared_ptr<const Light> create(Allocator<> allocator, const Scene *scene,
                                                    const std::optional<Triangle> &triangle) = 0;
    };

} // namespace akari::render