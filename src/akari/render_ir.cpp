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
                                                           Allocator<> alloc, Float &max_radiance) {
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
                // max_radiance = hmax(vpl0.radiance);
                max_radiance = 1.0 / (light_pdf * sample.pdfPos);
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
                beta *= sample->f() * std::abs(glm::dot(si->ns, sample->wi)) / sample->pdf;
                ray = Ray(si->p, sample->wi, Eps / std::abs(glm::dot(si->ng, sample->wi)));
                if (depth > config.min_depth) {
                    Float continue_prob = std::min<Float>(1.0, hmax(beta)) * 0.95;
                    if (continue_prob > sampler.next1d()) {
                        beta *= (Spectrum(1.0 / continue_prob));
                    } else {
                        break;
                    }
                }
            }
            // spdlog::info("{}", vpls.size());
            return vpls;
        }
        static Float ir_sample_fraction(const BSDFClosure &closure) {
            if (closure.isa<SpecularReflection>() || closure.isa<SpecularTransmission>() ||
                closure.isa<FresnelSpecular>()) {
                return 0.0;
            }
            if (auto glossy_refl = closure.get<MicrofacetReflection>()) {
                return glm::smoothstep(0.1f, 0.6f, glossy_refl->roughness);
            }
            if (auto mix = closure.get<MixBSDF>()) {
                return (1.0f - mix->fraction) * ir_sample_fraction(*mix->bsdf_A) +
                       mix->fraction * ir_sample_fraction(*mix->bsdf_B);
            }
            return 1.0;
        }
        static Float mis_weight(Float pdf_A, Float pdf_B) {
            pdf_A *= pdf_A;
            pdf_B *= pdf_B;
            return pdf_A / (pdf_A + pdf_B);
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
            Float max_radiance = 0.0;
            auto vpls = ir::generate_vpls(config, scene, vpl_sampler, vpl_alloc, max_radiance);
            auto kernel = [&](ivec2 id, uint32_t tid) {
                Sampler &sampler = samplers[id.x + id.y * film.resolution().x];
                Spectrum L(0.0);
                // Spectrum beta(1.0);
                auto camera_sample = scene.camera->generate_ray(sampler.next2d(), sampler.next2d(), id);
                Ray ray = camera_sample.ray;
                auto alloc = Allocator<>(buffers[tid]);
                // auto rec_sampler = sampler;
#if 1
                auto estimate = [&](const Ray ray, Spectrum beta, const int depth,
                                    const std::optional<SurfaceInteraction> prev, const std::optional<Float> prev_b,
                                    const Float prev_bsdf_pdf, const BSDFType prev_bsdf_type, auto &&self) -> void {
                    if (depth >= config.max_depth)
                        return;
                    if (depth > config.min_depth) {
                        Float continue_prob = std::min<Float>(1.0, hmax(beta)) * 0.95;
                        if (continue_prob > sampler.next1d()) {
                            beta *= (Spectrum(1.0 / continue_prob));
                        } else {
                            return;
                        }
                    }
                    auto si = scene.intersect(ray);
                    if (!si)
                        return;
                    if (prev_b) {
                        auto w0 = prev->p - si->p;
                        const auto dist_sqr = dot(w0, w0);
                        const auto w = w0 / std::sqrt(dist_sqr);
                        const auto G0 = std::abs(dot(w, prev->ng) * dot(w, si->ng)) / dist_sqr;
                        if (G0 < *prev_b) {
                            return;
                        }
                    }
                    auto wo = -ray.d;
                    auto *material = si->material();
                    if (!material)
                        return;
                    auto bsdf = material->evaluate(sampler, alloc, *si);
                    if (si->triangle.light) {
                        auto light = si->triangle.light;
                        Spectrum I = beta * light->Le(wo, si->sp());
                        if (depth == 0 || BSDFType::Unset != (prev_bsdf_type & BSDFType::Specular)) {
                            L += I;
                        } else {
                            PointGeometry ref;
                            ref.n = prev->ng;
                            ref.p = prev->p;
                            auto light_pdf = light->pdf_incidence(ref, -wo) * scene.light_sampler->pdf(light);
                            Float weight_bsdf = ir::mis_weight(prev_bsdf_pdf, light_pdf);
                            L += weight_bsdf * I;
                        }
                    }
                    auto ir_frac = ir::ir_sample_fraction(bsdf.closure());
                    AKR_ASSERT(ir_frac >= 0.0 && ir_frac <= 1.0);
                    // Direct lighting
                    if (depth == 0 || ir_frac < 1.0 - 1e-3) {
                        auto [light, light_pdf] = scene.light_sampler->sample(sampler.next2d());
                        if (light) {
                            LightSampleContext light_ctx;
                            light_ctx.u = sampler.next2d();
                            light_ctx.p = si->p;
                            LightSample light_sample = light->sample_incidence(light_ctx);
                            if (light_sample.pdf > 0.0) {

                                light_pdf *= light_sample.pdf;
                                auto f = light_sample.I * bsdf.evaluate(wo, light_sample.wi)()*
                                         std::abs(dot(si->ns, light_sample.wi));
                                Float bsdf_pdf = bsdf.evaluate_pdf(wo, light_sample.wi);
                                Float weight = [&, light_pdf = light_pdf]() -> Float {
                                    if (depth == 0)
                                        return 1.0;
                                    return ir::mis_weight(light_pdf, bsdf_pdf);
                                }();
                                Spectrum color = f / light_pdf * weight;
                                Ray shadow_ray = light_sample.shadow_ray;
                                if (!is_black(color) && !scene.occlude(shadow_ray)) {
                                    L += color * beta;
                                }
                            }
                        }
                    }
                    auto beta_bsdf = beta * (1.0f - ir_frac);
                    beta *= ir_frac;
                    astd::pmr::vector<Float> ks(alloc), bs(alloc);
                    for (const auto &vpl : vpls) {
                        auto w0 = vpl.p - si->p;
                        const auto dist_sqr = dot(w0, w0);
                        const auto w = w0 / std::sqrt(dist_sqr);
                        const auto G0 = std::abs(dot(w, vpl.ng) * dot(w, si->ng)) / dist_sqr;
                        const auto f = bsdf.evaluate(wo, w)() * vpl.bsdf->evaluate(vpl.wo, -w)();
                        const auto c = max_radiance;
                        const auto b = c / hmax(f);
                        const auto G = std::min(G0, b);
                        const auto k = std::max<Float>((G0 - b) / G0, 0.0f);
                        Spectrum contribution(0.0);
                        contribution = vpl.radiance * G * f;
                        const Ray shadow_ray(si->p, w0, 0.01, 1.0f - ShadowEps);
                        if (!is_black(contribution) && !scene.occlude(shadow_ray)) {
                            L += contribution * beta;
                        }
                        ks.emplace_back(k);
                        bs.emplace_back(b);

#    ifdef VPL_ENABLE_RECURSION
                        if (k > 0.0) {
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
                            auto next_ray = Ray(si->p, sample->wi, Eps / std::abs(glm::dot(si->ng, sample->wi)),
                                                std::sqrt(1.0 / b) * 1.01);
                            self(next_ray, beta, depth + 1, si, b, sample->pdf, sample->type, self);
                        }
#    endif
                    }
#    ifndef VPL_ENABLE_RECURSION
                    ([&]() -> void {
                        Distribution1D dist(ks.data(), ks.size(), alloc);
                        if (dist.integral() == 0.0)
                            return;
                        const auto [idx, k_pdf] = dist.sample_discrete(sampler.next1d());
                        const auto k = ks[idx];
                        const auto b = bs[idx];
                        if (k == 0.0)
                            return;
                        BSDFSampleContext sample_ctx{sampler.next1d(), sampler.next2d(), wo};
                        auto sample = bsdf.sample(sample_ctx);
                        if (!sample) {
                            return;
                        }
                        AKR_ASSERT(sample->pdf >= 0.0f);
                        if (sample->pdf == 0.0f) {
                            return;
                        }
                        beta *= sample->f() * std::abs(glm::dot(si->ns, sample->wi)) / sample->pdf;
                        auto next_ray = Ray(si->p, sample->wi, Eps / std::abs(glm::dot(si->ng, sample->wi)),
                                            std::sqrt(1.0 / b) * 1.01);
                        self(next_ray, beta, depth + 1, si, b, sample->pdf, sample->type, self);
                    })();
#    endif
                    auto F = [&]() -> void {
                        if (ir_frac < 1.0 - 1e-3) {
                            BSDFSampleContext sample_ctx{sampler.next1d(), sampler.next2d(), wo};
                            auto sample = bsdf.sample(sample_ctx);
                            if (!sample) {
                                return;
                            }
                            AKR_ASSERT(sample->pdf >= 0.0f);
                            if (sample->pdf == 0.0f) {
                                return;
                            }
                            beta_bsdf *= sample->f() * std::abs(glm::dot(si->ns, sample->wi)) / sample->pdf;
                            auto next_ray = Ray(si->p, sample->wi, Eps / std::abs(glm::dot(si->ng, sample->wi)));
                            self(next_ray, beta_bsdf, depth + 1, si, std::nullopt, sample->pdf, sample->type, self);
                        }
                    };
                    F();
                };
                estimate(ray, Spectrum(1.0), 0, std::nullopt, std::nullopt, 0.0, BSDFType::Unset, estimate);
#endif
#if 0
                Spectrum beta(1.0);
                int depth = 0;
                BSDFType prev_bsdf_type = BSDFType::Unset;
                std::optional<SurfaceInteraction> prev_si = std::nullopt;
                Float G_bound = 0;
                while (depth < config.max_depth) {
                    auto si = scene.intersect(ray);
                    if (!si)
                        break;
                    if (prev_si) {
                        auto w0 = prev_si->p - si->p;
                        const auto dist_sqr = dot(w0, w0);
                        const auto w = w0 / std::sqrt(dist_sqr);
                        const auto G0 = std::abs(dot(w, prev_si->ng) * dot(w, si->ng)) / dist_sqr;
                        if (G0 < G_bound) {
                            break;
                        }
                    }
                    auto wo = -ray.d;
                    auto *material = si->material();
                    if (!material)
                        break;
                    auto bsdf = material->evaluate(sampler, alloc, *si);
                    if (si->triangle.light) {
                        auto light = si->triangle.light;
                        if (depth == 0 || BSDFType::Unset != (prev_bsdf_type & BSDFType::Specular)) {
                            Spectrum I = beta * light->Le(wo, si->sp());
                            L += I;
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
                    // struct NextDir {
                    //     Float k =0.0;
                    //     Vec3 wi;
                    //     Spectrum beta;
                    // };
                    astd::pmr::vector<Float> ks(alloc), bs(alloc);
                    for (const auto &vpl : vpls) {
                        auto w0 = vpl.p - si->p;
                        const auto dist_sqr = dot(w0, w0);
                        const auto w = w0 / std::sqrt(dist_sqr);
                        const auto G0 = std::abs(dot(w, vpl.ng) * dot(w, si->ng)) / dist_sqr;
                        const auto f = bsdf.evaluate(wo, w);
                        const auto c = max_radiance; // * 2.0f;
                        const auto b = c / std::fmax(0.1f, hmax(f));
                        const auto G = std::min(G0, b);
                        const auto k = std::max<Float>((G0 - b) / G0, 0.0f);
                        ks.emplace_back(k);
                        bs.emplace_back(b);
                        Spectrum contribution(0.0);
                        contribution = vpl.radiance * G * f * vpl.bsdf->evaluate(vpl.wo, -w);
                        const Ray shadow_ray(si->p, w0, 0.01, 1.0f - ShadowEps);
                        if (!is_black(contribution) && !scene.occlude(shadow_ray)) {
                            L += contribution * beta;
                        }
                    }
                    Distribution1D dist(ks.data(), ks.size(), alloc);
                    if (dist.integral() == 0.0)
                        break;
                    const auto [idx, k_pdf] = dist.sample_discrete(sampler.next1d());
                    const auto k = ks[idx];
                    const auto b = bs[idx];
                    G_bound = b;
                    BSDFSampleContext sample_ctx{sampler.next1d(), sampler.next2d(), wo};
                    auto sample = bsdf.sample(sample_ctx);
                    if (!sample) {
                        break;
                    }
                    AKR_ASSERT(sample->pdf >= 0.0f);
                    if (sample->pdf == 0.0f) {
                        break;
                    }
                    beta *= sample->f * std::abs(glm::dot(si->ns, sample->wi)) / sample->pdf * k / k_pdf;
                    ray = Ray(si->p, sample->wi, Eps / std::abs(glm::dot(si->ng, sample->wi)));
                    prev_bsdf_type = sample->type;
                    prev_si = si;
                    depth++;
                }
#endif
                buffers[tid]->release();
                L = clamp_zero(L);
                L = min(L, Spectrum(5.0));
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