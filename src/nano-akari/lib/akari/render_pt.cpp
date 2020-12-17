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
    namespace pt {
        struct DirectLighting {
            Ray shadow_ray;
            Spectrum color;
            Float pdf;
        };

        struct SurfaceVertex {
            SurfaceInteraction si;
            Vec3 wo;
            Ray ray;
            Spectrum beta;
            std::optional<BSDF> bsdf;
            Float pdf = 0.0;
            BSDFType sampled_lobe = BSDFType::Unset;
            // SurfaceVertex() = default;
            SurfaceVertex(const Vec3 &wo, const SurfaceInteraction si) : wo(wo), si(si) {}
            Vec3 p() const { return si.p; }
            Vec3 ng() const { return si.ng; }
        };
        struct PathVertex : Variant<SurfaceVertex> {
            using Variant::Variant;
            Vec3 p() const {
                return dispatch([](auto &&arg) { return arg.p(); });
            }
            Vec3 ng() const {
                return dispatch([](auto &&arg) { return arg.ng(); });
            }
            Float pdf() const {
                return dispatch([](auto &&arg) { return arg.pdf; });
            }
            BSDFType sampled_lobe() const {
                return dispatch([=](auto &&arg) -> BSDFType {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, SurfaceVertex>) {
                        return arg.sampled_lobe;
                    }
                    return BSDFType::Unset;
                });
            }
            const Light *light(const Scene *scene) const {
                return dispatch([=](auto &&arg) -> const Light * {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, SurfaceVertex>) {
                        return arg.si.light();
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
            Spectrum emitter_direct;
            Spectrum beta = Spectrum(1.0f);
            Allocator<> allocator;
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
            std::pair<const Light *, Float> select_light() noexcept {
                return scene->light_sampler->sample(sampler->next2d());
            }

            std::optional<DirectLighting>
            compute_direct_lighting(SurfaceVertex &vertex, const std::pair<const Light *, Float> &selected) noexcept {
                auto [light, light_pdf] = selected;
                if (light) {
                    auto &si = vertex.si;
                    DirectLighting lighting;
                    LightSampleContext light_ctx;
                    light_ctx.u = sampler->next2d();
                    light_ctx.p = si.p;
                    LightSample light_sample = light->sample_incidence(light_ctx);
                    if (light_sample.pdf <= 0.0)
                        return std::nullopt;
                    light_pdf *= light_sample.pdf;
                    auto f = light_sample.I * vertex.bsdf->evaluate(vertex.wo, light_sample.wi) *
                             std::abs(dot(si.ns, light_sample.wi));
                    Float bsdf_pdf = vertex.bsdf->evaluate_pdf(vertex.wo, light_sample.wi);
                    lighting.color = f / light_pdf * mis_weight(light_pdf, bsdf_pdf);
                    lighting.shadow_ray = light_sample.shadow_ray;
                    lighting.pdf = light_pdf;
                    return lighting;
                } else {
                    return std::nullopt;
                }
            }

            void on_miss(const Ray &ray, const std::optional<PathVertex> &prev_vertex) noexcept {
                // if (scene->envmap) {
                //     on_hit_light(scene->envmap.get(), -ray.d, ShadingPoint(), prev_vertex);
                // }
            }

            void accumulate_radiance(const Spectrum &r) { L += r; }

            void on_hit_light(const Light *light, const Vec3 &wo, const ShadingPoint &sp,
                              const std::optional<PathVertex> &prev_vertex) {
                Spectrum I = beta * light->Le(wo, sp);
                if (depth == 0 || BSDFType::Unset != (prev_vertex->sampled_lobe() & BSDFType::Specular)) {
                    if (depth == 0) {
                        emitter_direct = I;
                    }
                    accumulate_radiance(I);
                } else {
                    PointGeometry ref;
                    ref.n = prev_vertex->ng();
                    ref.p = prev_vertex->p();
                    auto light_pdf = light->pdf_incidence(ref, -wo) * scene->light_sampler->pdf(light);
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
            std::optional<SurfaceVertex> on_surface_scatter(const Vec3 &wo, SurfaceInteraction &si,
                                                            const std::optional<PathVertex> &prev_vertex) noexcept {
                auto *material = si.material();
                if (si.triangle.light) {
                    on_hit_light(si.triangle.light, wo, si.sp(), prev_vertex);
                    return std::nullopt;
                } else if (depth < max_depth) {
                    SurfaceVertex vertex(wo, si);
                    auto bsdf = material->evaluate(*sampler, allocator, si);
                    BSDFSampleContext sample_ctx{sampler->next1d(), sampler->next2d(), wo};
                    auto sample = bsdf.sample(sample_ctx);
                    if (!sample) {
                        return std::nullopt;
                    }
                    AKR_ASSERT(sample->pdf >= 0.0f);
                    if (sample->pdf == 0.0f) {
                        return std::nullopt;
                    }
                    vertex.bsdf = bsdf;
                    vertex.ray = Ray(si.p, sample->wi, Eps / std::abs(glm::dot(si.ng, sample->wi)));
                    vertex.beta = sample->f * std::abs(glm::dot(si.ns, sample->wi)) / sample->pdf;
                    vertex.pdf = sample->pdf;
                    vertex.sampled_lobe = sample->type;
                    return vertex;
                }
                return std::nullopt;
            }
            void run_megakernel(const Camera *camera, const ivec2 &p) noexcept {
                auto camera_sample = camera_ray(camera, p);
                Ray ray = camera_sample.ray;
                std::optional<PathVertex> prev_vertex;
                while (true) {
                    auto si = scene->intersect(ray);
                    if (!si) {
                        on_miss(ray, prev_vertex);
                        break;
                    }
                    auto wo = -ray.d;
                    auto vertex = on_surface_scatter(wo, *si, prev_vertex);
                    if (!vertex) {
                        break;
                    }
                    if ((vertex->sampled_lobe & BSDFType::Specular) == BSDFType::Unset) {
                        std::optional<DirectLighting> has_direct = compute_direct_lighting(*vertex, select_light());
                        if (has_direct) {
                            auto &direct = *has_direct;
                            if (!is_black(direct.color) && !scene->occlude(direct.shadow_ray)) {
                                accumulate_radiance(beta * direct.color);
                            }
                        }
                    }
                    accumulate_beta(vertex->beta);
                    depth++;
                    if (depth > min_depth) {
                        Float continue_prob = std::min<Float>(1.0, hmax(beta)) * 0.95;
                        if (continue_prob > sampler->next1d()) {
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
    } // namespace pt

    std::pair<Spectrum, Spectrum> render_pt_pixel_separete_emitter_direct(PTConfig config, Allocator<> allocator,
                                                                          const Scene &scene, Sampler &sampler,
                                                                          const vec2 &p_film) {
        pt::GenericPathTracer pt;
        pt.min_depth = config.min_depth;
        pt.max_depth = config.max_depth;
        pt.L = Spectrum(0.0);
        pt.beta = Spectrum(1.0);
        pt.sampler = &sampler;
        pt.scene = &scene;
        pt.allocator = allocator;
        pt.run_megakernel(&scene.camera.value(), p_film);
        AKR_ASSERT(hmax(pt.L) >= 0.0 && hmax(pt.emitter_direct) >= 0.0);
        return std::make_pair(pt.emitter_direct, pt.L);
    }
    Film render_pt(PTConfig config, const Scene &scene) {
        Film film(scene.camera->resolution());
        std::vector<astd::pmr::monotonic_buffer_resource *> buffers;
        for (size_t i = 0; i < thread::num_work_threads(); i++) {
            buffers.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::new_delete_resource()));
        }
        thread::parallel_for(thread::blocked_range<2>(film.resolution(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            Sampler sampler = config.sampler;
            sampler.set_sample_index(id.y * film.resolution().x + id.x);
            for (int s = 0; s < config.spp; s++) {
                sampler.start_next_sample();
                auto L = render_pt_pixel(config, Allocator<>(buffers[tid]), scene, sampler, id);
                buffers[tid]->release();
                film.add_sample(id, L, 1.0);
            }
        });
        for (auto buf : buffers) {
            delete buf;
        }
        spdlog::info("render pt done");
        return film;
    }
} // namespace akari::render