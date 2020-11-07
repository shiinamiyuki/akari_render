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
#include <akari/core/variant.h>
#include <akari/render/scenegraph.h>
#include <akari/render/common.h>
#include <akari/core/memory.h>
#include <akari/render/sampler.h>
#include <akari/render/texture.h>
#include <akari/render/closure.h>
#include <optional>

namespace akari::render {
    class AKR_EXPORT PhaseFunction {
      public:
        virtual Float evaluate(const Vec3 &wo, const Vec3 &wi) const = 0;
        virtual std::pair<Vec3, Float> sample(const Vec2 &u, const Vec3 &wo) const = 0;
    };
    class AKR_EXPORT HGPhaseFunction : public PhaseFunction {
        const Float g;

      public:
        HGPhaseFunction(Float g) : g(g) {}
        Float evaluate(const Vec3 &wo, const Vec3 &wi) const override {
            const auto cosTheta = dot(wo, wi);
            const Float denom = 1 + g * g + 2 * g * cosTheta;
            return Inv4Pi * (1 - g * g) / (denom * std::sqrt(denom));
        }
        std::pair<Vec3, Float> sample(const Vec2 &u, const Vec3 &wo) const override {
            Float cosTheta;
            if (std::abs(g) < 1e-3f) {
                cosTheta = 1 - 2 * u[0];
            } else {
                const Float g2 = g * g;
                const Float sq = (1 - g2) / (1 - g + 2 * g * u[0]);
                cosTheta = (1 + g2 - sq * sq) / (2 * g);
            }
            const Float sinTheta = std::sqrt(std::max(0.0f, 1 - cosTheta * cosTheta));
            const Float phi = 2 * u[1] * Pi;
            Vec3 v1, v2;
            Frame frame(wo);

            auto w = spherical_to_xyz(sinTheta, cosTheta, phi);
            auto wi = frame.local_to_world(w);
            return std::make_pair(wi, evaluate(wo, wi));
        }
    };
    class Medium {
      public:
        virtual const PhaseFunction *evaluate(const Vec3 &p) = 0;
        virtual Spectrum tr(const Ray &ray, Sampler *sampler) = 0;
    };
    class MediumNode : public SceneGraphNode {
      public:
        virtual std::shared_ptr<const MediumNode> create_medium(Allocator<>) = 0;
    };
} // namespace akari::render