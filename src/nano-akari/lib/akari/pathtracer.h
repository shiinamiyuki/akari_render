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

#pragma once

#include <akari/util.h>
#include <akari/render.h>

namespace akari::render ::pt {
    struct DirectLighting {
        Ray shadow_ray;
        Vec3 wi = Vec3(0.0);
        BSDFValue radiance = BSDFValue::zero();
        Spectrum throughput = Spectrum(0.0);
        Float pdf = 0.0;
    };

    struct HitLight {
        Vec3 wo;
        const Light *light = nullptr;
        Spectrum I = Spectrum(0.0);
    };

    struct SurfaceVertex {
        SurfaceInteraction si;
        Vec3 wo;
        Ray ray;
        BSDFValue beta;
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
    struct PathTracerBase {
        const Scene *scene = nullptr;
        Sampler *sampler = nullptr;
        Spectrum L = Spectrum(0.0);
        Spectrum emitter_direct;
        Spectrum beta = Spectrum(1.0f);
        Allocator<> allocator;
        int depth = 0;
        int min_depth = 5;
        int max_depth = 5;
        PathTracerBase(const Scene *scene, Sampler *sampler, Allocator<> alloc, int min_depth, int max_depth)
            : scene(scene), sampler(sampler), allocator(alloc), min_depth(min_depth), max_depth(max_depth) {}
    };
    // Basic Path Tracing
    /*
    class PathVisitor {
    public:
        explicit PathVisitor(PathTracerBase *);

        // return true for accepting the scattering
        bool on_scatter(const PathVertex & cur, const std::optional<PathVertex> &prev);
        bool on_hit_light(const std::optional<PathVertex> &prev, const HitLight & hit);
        bool on_direct_lighting(const PathVertex & cur, const DirectLighting & direct);
        bool on_miss(const std::optional<PathVertex> &prev, const Ray & ray);
        bool on_advance_path(const PathVertex &cur, const Ray &ray);
    };
    */
    class NullPathVisitor {
      public:
        explicit NullPathVisitor(PathTracerBase *) {}

        // return true for accepting the scattering
        bool on_scatter(const PathVertex &cur, const std::optional<PathVertex> &prev) { return true; }
        bool on_direct_lighting(const PathVertex &cur, const DirectLighting &direct) { return true; }
        bool on_miss(const std::optional<PathVertex> &prev, const Ray &ray) { return true; }
        bool on_hit_light(const std::optional<PathVertex> &prev, const HitLight &hit) { return true; }
        bool on_advance_path(const PathVertex &cur, const Ray &ray) { return true; }
    };

    class SeparateEmitPathVisitor {
        PathTracerBase *pt;

      public:
        Spectrum emitter_direct;
        explicit SeparateEmitPathVisitor(PathTracerBase *pt) : pt(pt) {}

        // return true for accepting the scattering
        bool on_scatter(const PathVertex &cur, const std::optional<PathVertex> &prev) { return true; }
        bool on_direct_lighting(const PathVertex &cur, const DirectLighting &direct) { return true; }
        bool on_miss(const std::optional<PathVertex> &prev, const Ray &ray) { return true; }
        bool on_hit_light(const std::optional<PathVertex> &prev, const HitLight &hit) {
            if (pt->depth == 0) {
                emitter_direct = hit.I;
            }
            return true;
        }
        bool on_advance_path(const PathVertex &cur, const Ray &ray) { return true; }
    };
    enum class AOVKind {
        Emission,
        Normal,
        DiffuseAlbedo,
        DiffuseDirect,
        DiffuseIndirect,
        Specular, // Specular + Glossy
        NAOVKind
    };
    struct AOVs : std::array<Spectrum, (int)AOVKind::NAOVKind> {
        using Base = std::array<Spectrum, (int)AOVKind::NAOVKind>;
        AOVs() {
            for (auto &s : *this) {
                s = Spectrum(0.0);
            }
        }
        Spectrum &operator[](AOVKind kind) { return static_cast<Base &>(*this)[(int)kind]; }
    };

    template <class PathVisitor = NullPathVisitor, bool ComputeAOV = false>
    class GenericPathTracer : public PathTracerBase {
      public:
        AOVs aovs;
        Spectrum beta_diffuse = Spectrum(0.0);
        PathVisitor visitor;
        GenericPathTracer(const Scene *scene, Sampler *sampler, Allocator<> alloc, int min_depth, int max_depth)
            : PathTracerBase(scene, sampler, alloc, min_depth, max_depth), visitor(this) {}
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
                auto f = vertex.bsdf->evaluate(vertex.wo, light_sample.wi);
                Float bsdf_pdf = vertex.bsdf->evaluate_pdf(vertex.wo, light_sample.wi);
                lighting.throughput = light_sample.I * std::abs(dot(si.ns, light_sample.wi)) / light_pdf *
                                      mis_weight(light_pdf, bsdf_pdf);
                lighting.radiance = f * lighting.throughput;
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf = light_pdf;
                lighting.wi = light_sample.wi;
                if (visitor.on_direct_lighting(vertex, lighting)) {
                    return lighting;
                }
                return std::nullopt;
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
                ;
            } else {
                PointGeometry ref;
                ref.n = prev_vertex->ng();
                ref.p = prev_vertex->p();
                auto light_pdf = light->pdf_incidence(ref, -wo) * scene->light_sampler->pdf(light);
                if ((prev_vertex->sampled_lobe() & BSDFType::Specular) == BSDFType::Unset) {
                    Float weight_bsdf = mis_weight(prev_vertex->pdf(), light_pdf);
                    I *= weight_bsdf;
                }
            }
            HitLight hit{wo, light, I};
            if (visitor.on_hit_light(prev_vertex, hit)) {
                accumulate_radiance(I);
                if (depth == 0) {
                    aovs[AOVKind::DiffuseDirect] += I;
                }
            }
        }
        void accumulate_beta(const Spectrum &k) {
            beta *= k;
            if constexpr (ComputeAOV) {
                beta_diffuse *= k;
            }
        }
        
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
                vertex.beta = sample->f * (std::abs(glm::dot(si.ns, sample->wi)) / sample->pdf);
                vertex.pdf = sample->pdf;
                vertex.sampled_lobe = sample->type;
                return vertex;
            }
            return std::nullopt;
        }
        void run_megakernel(Ray ray, std::optional<PathVertex> prev_vertex) noexcept {

            while (true) {
                auto si = scene->intersect(ray);
                if (!si) {
                    on_miss(ray, prev_vertex);
                    break;
                }

                auto wo = -ray.d;
                auto vertex = on_surface_scatter(wo, *si, prev_vertex);
                if (!vertex || !visitor.on_scatter(*vertex, prev_vertex)) {
                    break;
                }
                if constexpr (ComputeAOV) {
                    if (depth == 0) {
                        aovs[AOVKind::Normal] = si->ns;
                        aovs[AOVKind::DiffuseAlbedo] = vertex->bsdf->closure().albedo().diffuse;
                    }
                }
                if ((vertex->sampled_lobe & BSDFType::Specular) == BSDFType::Unset) {
                    std::optional<DirectLighting> has_direct = compute_direct_lighting(*vertex, select_light());
                    if (has_direct) {
                        auto &direct = *has_direct;
                        if (!is_black(direct.radiance()) && !scene->occlude(direct.shadow_ray)) {
                            accumulate_radiance(beta * direct.radiance());
                            if constexpr (ComputeAOV) {
                                if (depth == 0)
                                    aovs[AOVKind::DiffuseDirect] += beta * direct.radiance.diffuse;
                                else {
                                    aovs[AOVKind::DiffuseIndirect] += beta_diffuse * direct.radiance();
                                }
                            }
                        }
                    }
                }
                accumulate_beta(vertex->beta());
                if constexpr (ComputeAOV) {
                    if (depth == 0) {
                        beta_diffuse = vertex->beta.diffuse;
                    }
                }
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
                if (!visitor.on_advance_path(*vertex, ray)) {
                    break;
                }
                prev_vertex = PathVertex(*vertex);
            }
            if constexpr (ComputeAOV) {
                aovs[AOVKind::Specular] = pt.L - (aovs[AOVKind::DiffuseDirect] + aovs[AOVKind::DiffuseIndirect]);
            }
        }
        void run_megakernel(const Camera *camera, const ivec2 &p) noexcept {
            auto camera_sample = camera_ray(camera, p);
            Ray ray = camera_sample.ray;
            run_megakernel(ray, std::nullopt);
        }
    };

} // namespace akari::render::pt

namespace akari::render {
    Spectrum pt_estimator(PTConfig config, const Scene &scene, Allocator<> alloc, Sampler &sampler, Ray &ray,
                          std::optional<pt::PathVertex> prev_vertex) {
        pt::GenericPathTracer<> pt(&scene, &sampler, alloc, config.min_depth, config.max_depth);
        // pt.min_depth = config.min_depth;
        // pt.max_depth = config.max_depth;
        // pt.L = Spectrum(0.0);
        // pt.beta = Spectrum(1.0);
        // pt.sampler = &sampler;
        // pt.scene = &scene;
        // pt.allocator = alloc;
        pt.run_megakernel(ray, prev_vertex);
        return pt.L;
    }
} // namespace akari::render