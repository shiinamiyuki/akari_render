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

#include <akari/kernel/sampler.h>
#include <akari/kernel/camera.h>
#include <akari/kernel/interaction.h>
#include <akari/kernel/material.h>
#include <akari/kernel/light.h>
#include <akari/kernel/scene.h>
namespace akari {
    AKR_VARIANT
    struct SurfaceHit {
        AKR_IMPORT_TYPES()
        Point2f uv;
        Vector3f wo;
        int geom_id = -1;
        int prim_id = -1;
        const Material<C> *material = nullptr;
        SurfaceHit() = default;
        SurfaceHit(const Ray3f &ray, const Intersection<C> &isct)
            : uv(isct.uv), wo(-ray.d), geom_id(isct.geom_id), prim_id(isct.prim_id) {}
    };
    AKR_VARIANT
    struct ScatteringEvent {
        AKR_IMPORT_TYPES()
        Ray3f ray;
        Spectrum beta;
        Float pdf;
    };
    template <class C>
    class GenericPathTracer {
      public:
        AKR_IMPORT_TYPES()
        Sampler<C> sampler;
        Spectrum L;
        Spectrum beta = Spectrum(1.0f);
        int depth = 0;
        int max_depth = 5;

        AKR_XPU CameraSample<C> camera_ray(const Camera<C> &camera, const Point2i &p) {
            CameraSample<C> sample = camera.generate_ray(sampler.next2d(), sampler.next2d(), p);
            return sample;
        }
        AKR_XPU astd::pair<const Light<C> *, Float> select_light(const Scene<C> &scene) {
            return scene.select_light(sampler.next2d());
        }

        AKR_XPU astd::optional<DirectLighting<C>>
        compute_direct_lighting(SurfaceInteraction<C> &si, const SurfaceHit<C> &surface_hit,
                                const astd::pair<const Light<C> *, Float> &selected) {
            auto [light, light_pdf] = selected;
            if (light) {
                DirectLighting<C> lighting;
                LightSampleContext<C> light_ctx;
                light_ctx.u = sampler.next2d();
                light_ctx.p = si.p;
                LightSample<C> light_sample = light->sample(light_ctx);
                light_pdf *= light_sample.pdf;
                auto f = light_sample.L * si.bsdf.evaluate(surface_hit.wo, light_sample.wi) *
                         std::abs(dot(si.ns, light_sample.wi));
                lighting.color = beta * f / light_pdf;
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf = light_pdf;
                return lighting;
            } else {
                return astd::nullopt;
            }
        }

        AKR_XPU void on_miss(const Scene<C> &scene, const Ray3f &ray) {}

        // @param mat_pdf: supplied if material is already chosen
        AKR_XPU astd::optional<ScatteringEvent<C>> on_surface_scatter(SurfaceInteraction<C> &si,
                                                                      const SurfaceHit<C> &surface_hit,
                                                                      astd::optional<Float> mat_pdf = astd::nullopt) {
            auto *material = surface_hit.material;
            auto wo = surface_hit.wo;
            MaterialEvalContext<C> ctx(sampler, si);
            if (material->template isa<EmissiveMaterial<C>>()) {
                if (depth == 0) {
                    auto *emission = material->template get<EmissiveMaterial<C>>();
                    bool face_front = dot(-wo, si.ng) < 0.0f;
                    if (emission->double_sided || face_front) {
                        L += beta * emission->color->evaluate(ctx.texcoords);
                        return astd::nullopt;
                    }
                }
            } else if (depth < max_depth) {
                ScatteringEvent<C> event;
                if (mat_pdf) {
                    auto pdf = mat_pdf.value();
                    si.bsdf = Material<C>::get_bsdf(astd::pair<const Material<C>*, Float>{material, pdf}, ctx);
                } else {
                    si.bsdf = material->get_bsdf(ctx);
                }
                BSDFSampleContext<C> sample_ctx(sampler.next2d(), wo);
                auto sample = si.bsdf.sample(sample_ctx);
                AKR_ASSERT(sample.pdf >= 0.0f);
                if (sample.pdf == 0.0f) {
                    return astd::nullopt;
                }
                event.ray = Ray3f(si.p, sample.wi, Constants<Float>::Eps() / std::abs(dot(si.ng, sample.wi)));
                event.beta = sample.f * std::abs(dot(si.ng, sample.wi)) / sample.pdf;
                event.pdf = sample.pdf;
                return event;
            }
            return astd::nullopt;
        }
    };
} // namespace akari