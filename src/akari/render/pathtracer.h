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
#include <variant>
#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/interaction.h>
#include <akari/render/mesh.h>
#include <akari/render/light.h>
#include <akari/render/common.h>

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

    struct SurfaceVertex {
        Triangle triangle;
        SurfaceHit surface_hit;
        Ray ray;
        Spectrum beta;
        BSDF bsdf;
        Float pdf = 0.0;
        BSDFType sampled_lobe = BSDFType::Unset;
        SurfaceVertex() = default;
        AKR_XPU SurfaceVertex(const Triangle &triangle, const SurfaceHit &surface_hit)
            : triangle(triangle), surface_hit(surface_hit) {}
        AKR_XPU Vec3 p() const { return triangle.p(surface_hit.uv); }
        AKR_XPU Vec3 ng() const { return triangle.ng(); }
    };
    struct PathVertex : Variant<SurfaceVertex> {
        using Variant::Variant;
        AKR_XPU Vec3 p() const {
            return dispatch([](auto &&arg) { return arg.p(); });
        }
        AKR_XPU Vec3 ng() const {
            return dispatch([](auto &&arg) { return arg.ng(); });
        }
        AKR_XPU Float pdf() const {
            return dispatch([](auto &&arg) { return arg.pdf; });
        }
        AKR_XPU BSDFType sampled_lobe() const {
            return dispatch([=](auto &&arg) -> BSDFType {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SurfaceVertex>) {
                    return arg.sampled_lobe;
                }
                return BSDFType::Unset;
            });
        }
        AKR_XPU const Light *light(const Scene *scene) const {
            return dispatch([=](auto &&arg) -> const Light * {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SurfaceVertex>) {
                    return scene->get_light(arg.surface_hit.geom_id, arg.surface_hit.prim_id);
                }
                return nullptr;
            });
        }
    };

    // Basic Path Tracing
    class GenericPathTracer {
      public:
        const Scene *scene = nullptr;
        Sampler *sampler = nullptr;
        Spectrum L;
        Spectrum beta = Spectrum(1.0f);
        Allocator<> *allocator = nullptr;
        int depth = 0;
        int min_depth = 5;
        int max_depth = 5;

        static Float mis_weight(Float pdf_A, Float pdf_B) {
            pdf_A *= pdf_A;
            pdf_B *= pdf_B;
            return pdf_A / (pdf_A + pdf_B);
        }
        CameraSample camera_ray(const Camera *camera, const ivec2 &p) noexcept {
            CameraSample sample = camera->generate_ray(sampler->next2d(), sampler->next2d(), p);
            return sample;
        }
        astd::pair<const Light *, Float> select_light() noexcept { return scene->select_light(sampler->next2d()); }

        astd::optional<DirectLighting>
        compute_direct_lighting(SurfaceInteraction &si, const SurfaceHit &surface_hit,
                                const astd::pair<const Light *, Float> &selected) noexcept {
            auto [light, light_pdf] = selected;
            if (light) {
                DirectLighting lighting;
                LightSampleContext light_ctx;
                light_ctx.u = sampler->next2d();
                light_ctx.p = si.p;
                LightSample light_sample = light->sample_incidence(light_ctx);
                if (light_sample.pdf <= 0.0)
                    return astd::nullopt;
                light_pdf *= light_sample.pdf;
                auto f = light_sample.I * si.bsdf.evaluate(surface_hit.wo, light_sample.wi) *
                         std::abs(dot(si.ns, light_sample.wi));
                Float bsdf_pdf = si.bsdf.evaluate_pdf(surface_hit.wo, light_sample.wi);
                lighting.color = f / light_pdf * mis_weight(light_pdf, bsdf_pdf);
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf = light_pdf;
                return lighting;
            } else {
                return astd::nullopt;
            }
        }

        void on_miss(const Ray &ray, const astd::optional<PathVertex> &prev_vertex) noexcept {
            if (scene->envmap) {
                on_hit_light(scene->envmap, -ray.d, ShadingPoint(), prev_vertex);
            }
        }

        void accumulate_radiance(const Spectrum &r) { L += r; }

        void on_hit_light(const Light *light, const Vec3 &wo, const ShadingPoint &sp,
                          const astd::optional<PathVertex> &prev_vertex) {
            Spectrum I = beta * light->Le(wo, sp);
            if (depth == 0) {
                accumulate_radiance(I);
            } else {
                vec3 prev_p = prev_vertex->p();
                ReferencePoint ref;
                ref.ng = prev_vertex->ng();
                ref.p = prev_vertex->p();
                auto light_pdf = light->pdf_incidence(ref, -wo) * scene->pdf_light(light);
                if ((prev_vertex->sampled_lobe() & BSDFType::Specular) != BSDFType::Unset) {
                    accumulate_radiance(I);
                } else {
                    Float weight_bsdf = mis_weight(prev_vertex->pdf(), light_pdf);
                    accumulate_radiance(weight_bsdf * I);
                }
            }
        }
        void accumulate_beta(const Spectrum &k) { beta *= k; }
        // @param mat_pdf: supplied if material is already chosen
        astd::optional<SurfaceVertex> on_surface_scatter(SurfaceInteraction &si, const SurfaceHit &surface_hit,
                                                         const astd::optional<PathVertex> &prev_vertex) noexcept {
            auto *material = surface_hit.material;
            auto &wo = surface_hit.wo;
            MaterialEvalContext ctx = si.mat_eval_ctx(allocator, sampler);
            if (material->is_emissive()) {
                auto light = scene->get_light(surface_hit.geom_id, surface_hit.prim_id);
                on_hit_light(light, wo, ctx.sp, prev_vertex);
                return astd::nullopt;
            } else if (depth < max_depth) {
                SurfaceVertex vertex(si.triangle, surface_hit);
                si.bsdf = material->get_bsdf(ctx);

                BSDFSampleContext sample_ctx(sampler->next2d(), wo);
                auto sample = si.bsdf.sample(sample_ctx);
                AKR_ASSERT(sample.pdf >= 0.0f);
                if (sample.pdf == 0.0f) {
                    return astd::nullopt;
                }
                vertex.bsdf = si.bsdf;
                vertex.ray = Ray(si.p, sample.wi, Eps / std::abs(glm::dot(si.ng, sample.wi)));
                vertex.beta = sample.f * std::abs(glm::dot(si.ng, sample.wi)) / sample.pdf;
                vertex.pdf = sample.pdf;
                return vertex;
            }
            return astd::nullopt;
        }
        void run_megakernel(const Camera *camera, const ivec2 &p) noexcept {
            auto camera_sample = camera_ray(camera, p);
            Ray ray = camera_sample.ray;
            astd::optional<PathVertex> prev_vertex;
            while (true) {
                auto hit = scene->intersect(ray);
                if (!hit) {
                    on_miss(ray, prev_vertex);
                    break;
                }
                SurfaceHit surface_hit(ray, *hit);
                auto trig = scene->get_triangle(surface_hit.geom_id, surface_hit.prim_id);
                surface_hit.material = trig.material;

                SurfaceInteraction si(surface_hit.uv, trig);
                auto vertex = on_surface_scatter(si, surface_hit, prev_vertex);
                if (!vertex) {
                    break;
                }
                astd::optional<DirectLighting> has_direct = compute_direct_lighting(si, surface_hit, select_light());
                if (has_direct) {
                    auto &direct = *has_direct;
                    if (!is_black(direct.color) && !scene->occlude(direct.shadow_ray)) {
                        accumulate_radiance(beta * direct.color);
                    }
                }
                accumulate_beta(vertex->beta);
                depth++;
                if (depth > min_depth) {
                    Float continue_prob = std::min<Float>(1.0, hmax(beta)) * 0.95;
                    if (continue_prob < sampler->next1d()) {
                        accumulate_beta(Spectrum(1.0 / continue_prob));
                    } else {
                        break;
                    }
                }
                ray = vertex->ray;
                prev_vertex = PathVertex(*vertex);
            }
        }
    };
} // namespace akari::render