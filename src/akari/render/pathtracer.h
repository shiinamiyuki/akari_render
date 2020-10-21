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
// OUT OF OR IN CON

#pragma once

#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/light.h>
#include <akari/shaders/common.h>

namespace akari::render {
    struct SurfaceHit {
        vec2 uv;
        Vec3 wo;
        int geom_id = -1;
        int prim_id = -1;
        const Material *material = nullptr;
        SurfaceHit() = default;
        SurfaceHit(const Ray &ray, const Intersection &isct)
            : uv(isct.uv), wo(-ray.d), geom_id(isct.geom_id), prim_id(isct.prim_id) {}
    };

    struct ScatteringEvent {
        Ray ray;
        Spectrum beta;
        Float pdf;
    };

    template <typename Derived>
    class GenericPathTracer {
      public:
        const Scene *scene = nullptr;
        Sampler *sampler = nullptr;
        Spectrum L;
        Spectrum beta = Spectrum(1.0f);
        Allocator<> *allocator = nullptr;
        int depth = 0;
        int max_depth = 5;
        Derived &derived() noexcept { return static_cast<Derived &>(*this); }
        const Derived &derived() const noexcept { return static_cast<const Derived &>(*this); }
        CameraSample camera_ray(const Camera *camera, const ivec2 &p) noexcept {
            CameraSample sample = camera->generate_ray(sampler->next2d(), sampler->next2d(), p);
            return sample;
        }
        std::pair<const Light *, Float> select_light() noexcept { return scene->select_light(sampler->next2d()); }

        std::optional<DirectLighting>
        compute_direct_lighting(SurfaceInteraction &si, const SurfaceHit &surface_hit,
                                const std::pair<const Light *, Float> &selected) noexcept {
            auto [light, light_pdf] = selected;
            if (light) {
                DirectLighting lighting;
                LightSampleContext light_ctx;
                light_ctx.u = sampler->next2d();
                light_ctx.p = si.p;
                LightSample light_sample = light->sample(light_ctx);
                if (light_sample.pdf <= 0.0)
                    return std::nullopt;
                light_pdf *= light_sample.pdf;
                auto f = light_sample.L * si.bsdf->evaluate(surface_hit.wo, light_sample.wi) *
                         std::abs(dot(si.ns, light_sample.wi));
                lighting.color = beta * f / light_pdf;
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf = light_pdf;
                return lighting;
            } else {
                return std::nullopt;
            }
        }

        void on_miss(const Ray &ray) noexcept { derived()._on_miss(ray); }

        // @param mat_pdf: supplied if material is already chosen
        std::optional<ScatteringEvent> on_surface_scatter(SurfaceInteraction &si,
                                                          const SurfaceHit &surface_hit) noexcept {
            auto *material = surface_hit.material;
            auto wo = surface_hit.wo;
            MaterialEvalContext ctx(allocator, sampler, si);
            if (material->is_emissive()) {

                if (depth == 0) {
                    auto *emission = material->as_emissive();
                    bool face_front = dot(-wo, si.ng) < 0.0f;
                    if (emission->double_sided || face_front) {
                        L += beta * emission->color->evaluate(ctx.texcoords);
                        return std::nullopt;
                    }
                }
            } else if (depth < max_depth) {
                ScatteringEvent event;
                si.bsdf = allocator->new_object<BSDF>(material->get_bsdf(ctx));

                BSDFSampleContext sample_ctx(sampler->next2d(), wo);
                auto sample = si.bsdf->sample(sample_ctx);
                AKR_ASSERT(sample.pdf >= 0.0f);
                if (sample.pdf == 0.0f) {
                    return std::nullopt;
                }
                event.ray = Ray(si.p, sample.wi, Eps / std::abs(glm::dot(si.ng, sample.wi)));
                event.beta = sample.f * std::abs(glm::dot(si.ng, sample.wi)) / sample.pdf;
                event.pdf = sample.pdf;
                return event;
            }
            return std::nullopt;
        }
        void run_megakernel(const Camera *camera, const ivec2 &p) noexcept {
            auto camera_sample = camera_ray(camera, p);
            Ray ray = camera_sample.ray;
            while (true) {
                auto hit = scene->intersect(ray);
                if (!hit) {
                    on_miss(ray);
                    break;
                }
                SurfaceHit surface_hit(ray, *hit);
                auto trig = scene->get_triangle(surface_hit.geom_id, surface_hit.prim_id);
                surface_hit.material = trig.material;
                SurfaceInteraction si(surface_hit.uv, trig);

                auto has_event = on_surface_scatter(si, surface_hit);
                if (!has_event) {
                    break;
                }
                std::optional<DirectLighting> has_direct = compute_direct_lighting(si, surface_hit, select_light());
                if (has_direct) {
                    auto &direct = *has_direct;
                    if (!is_black(direct.color) && !scene->occlude(direct.shadow_ray)) {
                        L += direct.color;
                    }
                }
                auto event = *has_event;
                beta *= event.beta;
                depth++;
                ray = event.ray;
            }
        }
    };
} // namespace akari::render