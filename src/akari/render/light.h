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
    };
    class AreaLight final : public Light {
      public:
        const Texture *color;
        bool double_sided = false;
        Triangle triangle;
        vec3 ng;
        AreaLight() = default;
        AreaLight(Triangle triangle) : triangle(triangle) {
            color = triangle.material->as_emissive()->color;
            double_sided = triangle.material->as_emissive()->double_sided;
            ng = triangle.ng();
        }
        Spectrum Le(const Vec3 &wo, const ShadingPoint &sp) const override {
            bool face_front = dot(wo, ng) > 0.0;
            if (double_sided || face_front) {
                return color->evaluate(sp);
            }
            return Spectrum(0.0);
        }
        Float pdf_incidence(const ReferencePoint &ref, const vec3 &wi) const override {
            Ray ray(ref.p, wi);
            auto hit = triangle.intersect(ray);
            if (!hit) {
                return 0.0f;
            }
            Float SA = triangle.area() * (-glm::dot(wi, triangle.ng())) / (hit->first * hit->first);
            return 1.0f / SA;
        }
        LightSample sample_incidence(const LightSampleContext &ctx) const override {
            auto coords = uniform_sample_triangle(ctx.u);
            auto p = triangle.p(coords);
            LightSample sample;
            sample.ng = triangle.ng();
            sample.wi = p - ctx.p;
            auto dist_sqr = dot(sample.wi, sample.wi);
            sample.wi /= sqrt(dist_sqr);
            sample.I = color->evaluate(triangle.texcoord(coords));
            sample.pdf = dist_sqr / max(Float(0.0), -dot(sample.wi, sample.ng)) / triangle.area();
            sample.shadow_ray = Ray(p, -sample.wi, Eps / std::abs(dot(sample.wi, sample.ng)),
                                    sqrt(dist_sqr) * (Float(1.0f) - ShadowEps));
            return sample;
        }
    };

    class AKR_EXPORT InfiniteAreaLight : public Light {
        Transform w2l, l2w;
        static constexpr size_t distribution_map_resolution = 4096;
        std::unique_ptr<Distribution2D> distribution = nullptr;
        Bounds3f world_bounds;
        const Texture *texture = nullptr;
        Float _power = 0.0;

      public:
        static vec2 get_uv(const vec3 &wi) {
            auto theta = spherical_theta(wi);
            // auto sinTheta = std::sin(theta);
            auto phi = spherical_phi(wi);
            return vec2(phi * Inv2Pi, 1.0f - theta * InvPi);
        }
        Float power() const { return _power; }
        Spectrum Le(const Vec3 &wo, const ShadingPoint &) const override {
            auto uv = get_uv(w2l.apply_vector(-wo));
            ShadingPoint sp;
            sp.texcoords = uv;
            return texture->evaluate(sp);
        }
        Float pdf_incidence(const ReferencePoint &ref, const vec3 &wi) const override {
            if (!distribution)
                return 0.0f;
            auto w = w2l.apply_vector(wi);
            auto theta = spherical_theta(w);
            auto sinTheta = std::sin(theta);
            auto phi = spherical_phi(w);
            if (sinTheta == 0.0f)
                return 0.0f;
            return distribution->pdf_continuous(vec2(phi * Inv2Pi, 1.0f - theta * InvPi)) / (2 * Pi * Pi * sinTheta);
        }
        LightSample sample_incidence(const LightSampleContext &ctx) const override {
            LightSample sample;
            Float mapPdf = 0.0f;
            if (!distribution) {
                mapPdf = 0.0f;
                sample.pdf = 0.0f;
                sample.I = Spectrum(0);
                return sample;
            }
            auto uv = distribution->sample_continuous(ctx.u, &mapPdf);
            if (mapPdf == 0.0f) {
                sample.I = Spectrum(0);
                sample.pdf = 0.0f;
                return sample;
            }
            auto theta = (1.0f - uv[1]) * Pi;
            auto phi = uv[0] * 2.0f * Pi;
            Float cosTheta = std::cos(theta);
            Float sinTheta = std::sin(theta);
            sample.wi = l2w.apply_vector(spherical_to_xyz(sinTheta, cosTheta, phi));
            if (sinTheta == 0.0f)
                sample.pdf = 0.0f;
            else
                sample.pdf = mapPdf / (2 * Pi * Pi * sinTheta);
            sample.I = texture->evaluate(ShadingPoint{uv});
            sample.shadow_ray = Ray(ctx.p, sample.wi, Eps, Inf);
            return sample;
        }
        static std::unique_ptr<InfiniteAreaLight> create(const Scene &scene, const TRSTransform &transform,
                                                         const Texture *texture, Allocator<>);
    };
} // namespace akari::render