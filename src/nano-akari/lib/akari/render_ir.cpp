// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <akari/util.h>
#include <akari/render.h>
#include <spdlog/spdlog.h>

namespace akari::render {
    namespace ir {
        struct VirtualPointLight {
            std::optional<BSDF> bsdf;
            vec3 wo;
            vec3 ng;
            vec3 p;
            Spectrum radiance = Spectrum(0.0);
            const Light *light = nullptr;
        };
        astd::pmr::vector<VirtualPointLight> generate_vpls(IRConfig config, const Scene &scene, Sampler &sampler,
                                                           Allocator<> alloc) {
            astd::pmr::vector<VirtualPointLight> vpls(alloc);
            Ray ray;
            Spectrum L(0.0);
            Spectrum beta(1.0);
            {
                VirtualPointLight vpl0;
                auto [light, light_pdf] = scene.light_sampler->sample(sampler.next2d());
                if (!light) {
                    return vpls;
                }
                vpl0.light = light;
                auto sample = light->sample_emission(sampler);
                vpl0.radiance = sample.E / (light_pdf * sample.pdfPos);
                vpl0.ng = sample.ng;
                vpl0.p = sample.ray.o;
                vpl0.bsdf = std::nullopt;
                L = vpl0.radiance;
                beta = Spectrum(1.0 / sample.pdfDir);
                ray = sample.ray;
                // vpls.emplace_back(vpl0);
            }
            for (int depth = 0; depth < config.max_depth; depth++) {
                auto si = scene.intersect(ray);
                if (!si)
                    break;
                auto wo = -ray.d;
                auto *material = si->material();
                if (!material)
                    break;
                auto bsdf = material->evaluate(sampler, alloc, *si);
                vpls.emplace_back(VirtualPointLight{bsdf, wo, si->ng, si->p, L * beta, nullptr});
                BSDFSampleContext sample_ctx{sampler.next1d(), sampler.next2d(), wo};
                auto sample = bsdf.sample(sample_ctx);
                if (!sample) {
                    break;
                }
                AKR_ASSERT(sample->pdf >= 0.0f);
                if (sample->pdf == 0.0f) {
                    break;
                }
                beta *= sample->f * std::abs(glm::dot(si->ns, sample->wi)) / sample->pdf;
                ray = Ray(si->p, sample->wi, Eps / std::abs(glm::dot(si->ng, sample->wi)));
            }
            // spdlog::info("{}", vpls.size());
            return vpls;
        }
    } // namespace ir
    Image render_ir(IRConfig config, const Scene &scene) {
        Film film(scene.camera->resolution());
        std::vector<astd::pmr::monotonic_buffer_resource *> buffers;
        for (size_t i = 0; i < thread::num_work_threads(); i++) {
            buffers.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::new_delete_resource()));
        }
        auto vpl_sampler = config.sampler;
        std::vector<Sampler> samplers(hprod(scene.camera->resolution()));
        for (size_t i = 0; i < samplers.size(); i++) {
            samplers[i] = config.sampler;
            samplers[i].set_sample_index(i);
        }
        astd::pmr::monotonic_buffer_resource vpl_buf(astd::pmr::new_delete_resource());
        for (uint32_t pass = 0; pass < config.spp; pass++) {
            vpl_sampler.start_next_sample();
            Allocator<> vpl_alloc(&vpl_buf);
            auto vpls = ir::generate_vpls(config, scene, vpl_sampler, vpl_alloc);
            auto kernel = [&](ivec2 id, uint32_t tid) {
                Sampler &sampler = samplers[id.x + id.y * film.resolution().x];
                Spectrum L(0.0);
                Spectrum beta(1.0);
                auto camera_sample = scene.camera->generate_ray(sampler.next2d(), sampler.next2d(), id);
                Ray ray = camera_sample.ray;
                auto alloc = Allocator<>(buffers[tid]);
                int depth = 0;
                while (depth < config.max_depth) {
                    BSDFType prev_bsdf_type = BSDFType::Unset;
                    auto si = scene.intersect(ray);
                    if (!si)
                        break;
                    auto wo = -ray.d;
                    auto *material = si->material();
                    if (!material)
                        break;
                    auto bsdf = material->evaluate(sampler, alloc, *si);
                    if (si->triangle.light) {
                        auto light = si->triangle.light;
                        if (depth == 0 || BSDFType::Unset != (prev_bsdf_type & BSDFType::Specular)) {
                            Spectrum I = beta * light->Le(wo, si->sp());
                            L += I * beta;
                        }
                    }
                    // Direct lighting
                    if (depth == 0) {
                        auto [light, light_pdf] = scene.light_sampler->sample(sampler.next2d());
                        if (light) {
                            LightSampleContext light_ctx;
                            light_ctx.u = sampler.next2d();
                            light_ctx.p = si->p;
                            LightSample light_sample = light->sample_incidence(light_ctx);
                            if (light_sample.pdf > 0.0) {

                                light_pdf *= light_sample.pdf;
                                auto f = light_sample.I * bsdf.evaluate(wo, light_sample.wi) *
                                         std::abs(dot(si->ns, light_sample.wi));
                                Float bsdf_pdf = bsdf.evaluate_pdf(wo, light_sample.wi);
                                Spectrum color = f / light_pdf;
                                Ray shadow_ray = light_sample.shadow_ray;
                                if (!is_black(color) && !scene.occlude(shadow_ray)) {
                                    L += color * beta;
                                }
                            }
                        }
                    }
                    for (const auto &vpl : vpls) {
                        auto w0 = vpl.p - si->p;
                        const auto dist_sqr = dot(w0, w0);
                        auto w = w0 / std::sqrt(dist_sqr);
                        const auto G = std::abs(dot(w, vpl.ng) * dot(w, si->ng)) / dist_sqr;
                        Spectrum contribution(0.0);

                        contribution = vpl.radiance * beta * G * bsdf.evaluate(wo, w) * vpl.bsdf->evaluate(vpl.wo, -w);

                        Ray shadow_ray(si->p, w0, 0.01, 1.0f - ShadowEps);
                        if (!is_black(contribution) && !scene.occlude(shadow_ray)) {
                            L += contribution;
                        }
                    }
                    depth++;
                    break;
                }
                buffers[tid]->release();
                film.add_sample(id, L, 1.0);
            };
            thread::parallel_for(thread::blocked_range<2>(film.resolution(), ivec2(16, 16)), kernel);
            vpl_buf.release();
        }
        for (auto buf : buffers) {
            delete buf;
        }
        spdlog::info("render ir done");
        return film.to_rgb_image();
    }
} // namespace akari::render