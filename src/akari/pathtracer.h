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

    template <class C, int N = 16>
    struct MediumStack : FixedVector<const Medium<CPU> *, N> {
        auto top() const { return this->back(); }
        // void update(const SurfaceInteraction & si, const astd::optional<MediumInteraction> & mi, const Ray & ray){
        //     if(empty() &&)
        // }
        void update(const SurfaceInteraction<CPU> &si, const Ray &ray) {
            if (!si.medium() || si.material()) {
                return;
            }
            bool going_in = dot(si.ng, ray.d) < 0.0;
            if (going_in) {
                if (this->size() < N - 1)
                    this->push_back(si.medium());
            } else {
                if (!this->empty())
                    this->pop_back();
            }
        }
    };
    struct DirectLighting {
        Ray shadow_ray;
        Vec3 wi             = Vec3(0.0);
        BSDFValue radiance  = BSDFValue::zero();
        Spectrum throughput = Spectrum(0.0);
        Float pdf           = 0.0;
    };

    struct VolumeDirectLighting {
        Ray shadow_ray;
        Vec3 wi           = Vec3(0.0);
        Spectrum radiance = Spectrum(0.0);
        Float pdf         = 0.0;
    };

    struct HitLight {
        Vec3 wo;
        const Light<CPU> *light = nullptr;
        Spectrum I              = Spectrum(0.0);
    };

    struct SurfaceVertex {
        Vec3 wo;
        SurfaceInteraction<CPU> si;
        Ray ray;
        BSDFValue beta;
        astd::optional<BSDF<CPU>> bsdf;
        Float pdf             = 0.0;
        BSDFType sampled_lobe = BSDFType::Unset;
        // SurfaceVertex() = default;
        SurfaceVertex(const Vec3 &wo, const SurfaceInteraction<CPU> &si) : wo(wo), si(si) {}
        Vec3 p() const { return si.p; }
        Vec3 ng() const { return si.ng; }
    };
    struct MediumVertex {
        Vec3 wo;
        MediumInteraction mi;
        Ray ray;
        Spectrum beta = Spectrum(1.0);
        Float pdf     = 0.0;
        explicit MediumVertex(const Vec3 &wo, const MediumInteraction &mi) : wo(wo), mi(mi) {}
        Vec3 p() const { return mi.p; }
        Vec3 ng() const { return Vec3(0); }
    };
    struct PathVertex : Variant<SurfaceVertex, MediumVertex> {
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
        const Light<CPU> *light(const Scene<CPU> *scene) const {
            return dispatch([=](auto &&arg) -> const Light<CPU> * {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, SurfaceVertex>) {
                    return arg.si.light();
                }
                return nullptr;
            });
        }
    };
    struct PathTracerBase {
        const Scene<CPU> *scene = nullptr;
        Sampler<CPU> *sampler   = nullptr;
        Spectrum L              = Spectrum(0.0);
        Spectrum emitter_direct;
        Spectrum beta = Spectrum(1.0f);
        Allocator<> allocator;
        int depth     = 0;
        int min_depth = 5;
        int max_depth = 5;
        PathTracerBase(const Scene<CPU> *scene, Sampler<CPU> *sampler, Allocator<> alloc, int min_depth, int max_depth)
            : scene(scene), sampler(sampler), allocator(alloc), min_depth(min_depth), max_depth(max_depth) {}
    };
    // Basic Path Tracing
    /*
    class PathVisitor {
    public:
        explicit PathVisitor(PathTracerBase *);

        // return true for accepting the scattering
        bool on_scatter(const PathVertex & cur, const astd::optional<PathVertex> &prev);
        bool on_hit_light(const astd::optional<PathVertex> &prev, const HitLight & hit);
        bool on_direct_lighting(const PathVertex & cur, const DirectLighting & direct);
        bool on_miss(const astd::optional<PathVertex> &prev, const Ray & ray);
        bool on_advance_path(const PathVertex &cur, const Ray &ray);
    };
    */
    class NullPathVisitor {
      public:
        explicit NullPathVisitor(PathTracerBase *) {}

        // return true for accepting the scattering
        bool on_scatter(const PathVertex &cur, const astd::optional<PathVertex> &prev) { return true; }
        bool on_direct_lighting(const PathVertex &cur, const DirectLighting &direct) { return true; }
        bool on_miss(const astd::optional<PathVertex> &prev, const Ray &ray) { return true; }
        bool on_hit_light(const astd::optional<PathVertex> &prev, const HitLight &hit) { return true; }
        bool on_advance_path(const PathVertex &cur, const Ray &ray) { return true; }
    };

    class SeparateEmitPathVisitor {
        PathTracerBase *pt;

      public:
        Spectrum emitter_direct;
        explicit SeparateEmitPathVisitor(PathTracerBase *pt) : pt(pt) {}

        // return true for accepting the scattering
        bool on_scatter(const PathVertex &cur, const astd::optional<PathVertex> &prev) { return true; }
        bool on_direct_lighting(const PathVertex &cur, const DirectLighting &direct) { return true; }
        bool on_miss(const astd::optional<PathVertex> &prev, const Ray &ray) { return true; }
        bool on_hit_light(const astd::optional<PathVertex> &prev, const HitLight &hit) {
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
    struct AOVs : astd::array<Spectrum, (int)AOVKind::NAOVKind> {
        using Base = astd::array<Spectrum, (int)AOVKind::NAOVKind>;
        AOVs() {
            for (auto &s : *this) {
                s = Spectrum(0.0);
            }
        }
        Spectrum &operator[](AOVKind kind) { return static_cast<Base &>(*this)[(int)kind]; }
    };
    static Float mis_weight(Float pdf_A, Float pdf_B) {
        pdf_A *= pdf_A;
        pdf_B *= pdf_B;
        return pdf_A / (pdf_A + pdf_B);
    }
    template <class PathVisitor = NullPathVisitor>
    class SimplePathTracer : public PathTracerBase {
      public:
        PathVisitor visitor;
        SimplePathTracer(const Scene<CPU> *scene, Sampler<CPU> *sampler, Allocator<> alloc, int min_depth,
                         int max_depth)
            : PathTracerBase(scene, sampler, alloc, min_depth, max_depth), visitor(this) {}

        CameraSample camera_ray(const Camera<CPU> *camera, const ivec2 &p) noexcept {
            CameraSample sample = camera->generate_ray(sampler->next2d(), sampler->next2d(), p);
            return sample;
        }
        std::pair<const Light<CPU> *, Float> select_light() noexcept {
            return scene->light_sampler->sample(sampler->next2d());
        }

        astd::optional<DirectLighting>
        compute_direct_lighting(SurfaceVertex &vertex, const std::pair<const Light<CPU> *, Float> &selected) noexcept {
            auto [light, light_pdf] = selected;
            if (light) {
                auto &si = vertex.si;
                DirectLighting lighting;
                LightSampleContext light_ctx;
                light_ctx.u              = sampler->next2d();
                light_ctx.p              = si.p;
                LightSample light_sample = light->sample_incidence(light_ctx);
                if (light_sample.pdf <= 0.0)
                    return astd::nullopt;
                light_pdf *= light_sample.pdf;
                auto f              = vertex.bsdf->evaluate(vertex.wo, light_sample.wi);
                Float bsdf_pdf      = vertex.bsdf->evaluate_pdf(vertex.wo, light_sample.wi);
                lighting.throughput = light_sample.I * std::abs(dot(si.ns, light_sample.wi)) / light_pdf *
                                      mis_weight(light_pdf, bsdf_pdf);
                lighting.radiance   = f * lighting.throughput;
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf        = light_pdf;
                lighting.wi         = light_sample.wi;
                if (visitor.on_direct_lighting(vertex, lighting)) {
                    return lighting;
                }
                return astd::nullopt;
            } else {
                return astd::nullopt;
            }
        }

        void on_miss(const Ray &ray, const astd::optional<PathVertex> &prev_vertex) noexcept {
            // if (scene->envmap) {
            //     on_hit_light(scene->envmap.get(), -ray.d, ShadingPoint(), prev_vertex);
            // }
        }

        void accumulate_radiance(const Spectrum &r) { L += r; }

        void on_hit_light(const Light<CPU> *light, const Vec3 &wo, const ShadingPoint &sp,
                          const astd::optional<PathVertex> &prev_vertex) {
            Spectrum I = beta * light->Le(wo, sp);
            if (depth == 0 || BSDFType::Unset != (prev_vertex->sampled_lobe() & BSDFType::Specular)) {
                ;
            } else {
                PointGeometry ref;
                ref.n          = prev_vertex->ng();
                ref.p          = prev_vertex->p();
                auto light_pdf = light->pdf_incidence(ref, -wo) * scene->light_sampler->pdf(light);
                if ((prev_vertex->sampled_lobe() & BSDFType::Specular) == BSDFType::Unset) {
                    Float weight_bsdf = mis_weight(prev_vertex->pdf(), light_pdf);
                    I *= weight_bsdf;
                }
            }
            HitLight hit{wo, light, I};
            if (visitor.on_hit_light(prev_vertex, hit)) {
                accumulate_radiance(I);
            }
        }
        void accumulate_beta(const Spectrum &k) { beta *= k; }

        astd::optional<SurfaceVertex> on_surface_scatter(const Vec3 &wo, SurfaceInteraction<CPU> &si,
                                                         const astd::optional<PathVertex> &prev_vertex) noexcept {
            auto *material = si.material();
            if (si.triangle.light) {
                on_hit_light(si.triangle.light, wo, si.sp(), prev_vertex);
                return astd::nullopt;
            } else if (depth < max_depth) {
                SurfaceVertex vertex(wo, si);
                auto bsdf = material->evaluate(*sampler, allocator, si);
                BSDFSampleContext sample_ctx{sampler->next1d(), sampler->next2d(), wo};
                auto sample = bsdf.sample(sample_ctx);
                if (!sample) {
                    return astd::nullopt;
                }
                AKR_ASSERT(sample->pdf >= 0.0f);
                if (sample->pdf == 0.0f) {
                    return astd::nullopt;
                }
                vertex.bsdf         = bsdf;
                vertex.ray          = Ray(si.p, sample->wi, Eps / std::abs(glm::dot(si.ng, sample->wi)));
                vertex.beta         = sample->f * (std::abs(glm::dot(si.ns, sample->wi)) / sample->pdf);
                vertex.pdf          = sample->pdf;
                vertex.sampled_lobe = sample->type;
                return vertex;
            }
            return astd::nullopt;
        }
        void run_megakernel(Ray ray, astd::optional<PathVertex> prev_vertex) noexcept {

            while (true) {
                auto si = scene->intersect(ray);
                if (!si) {
                    on_miss(ray, prev_vertex);
                    break;
                }

                auto wo     = -ray.d;
                auto vertex = on_surface_scatter(wo, *si, prev_vertex);
                if (!vertex || !visitor.on_scatter(*vertex, prev_vertex)) {
                    break;
                }
                if ((vertex->sampled_lobe & BSDFType::Specular) == BSDFType::Unset) {
                    astd::optional<DirectLighting> has_direct = compute_direct_lighting(*vertex, select_light());
                    if (has_direct) {
                        auto &direct = *has_direct;
                        if (!is_black(direct.radiance()) && !scene->occlude(direct.shadow_ray)) {
                            accumulate_radiance(beta * direct.radiance());
                        }
                    }
                }
                accumulate_beta(vertex->beta());

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
            L = clamp_zero(L);
        }
        void run_megakernel(const Camera<CPU> *camera, const ivec2 &p) noexcept {
            auto camera_sample = camera_ray(camera, p);
            Ray ray            = camera_sample.ray;
            run_megakernel(ray, astd::nullopt);
        }
    };

    template <class PathVisitor = NullPathVisitor>
    class UnifiedPathTracer : public PathTracerBase {
      public:
        struct Config {
            bool compute_aov = false;
            bool use_guiding = false;
            bool volumetric  = true;
            bool use_nee     = true;
        };
        Config config;
        PathVisitor visitor;
        UnifiedPathTracer(const Scene<CPU> *scene, Sampler<CPU> *sampler, Allocator<> alloc, int min_depth,
                          int max_depth)
            : PathTracerBase(scene, sampler, alloc, min_depth, max_depth), visitor(this) {}
        CameraSample camera_ray(const Camera<CPU> *camera, const ivec2 &p) noexcept {
            CameraSample sample = camera->generate_ray(sampler->next2d(), sampler->next2d(), p);
            return sample;
        }
        std::pair<const Light<CPU> *, Float> select_light() noexcept {
            return scene->light_sampler->sample(sampler->next2d());
        }
        astd::optional<VolumeDirectLighting>
        compute_direct_lighting(const MediumVertex &vertex,
                                const std::pair<const Light<CPU> *, Float> &selected) noexcept {
            auto [light, light_pdf] = selected;
            if (light) {
                auto &mi = vertex.mi;
                VolumeDirectLighting lighting;
                LightSampleContext light_ctx;
                light_ctx.u              = sampler->next2d();
                light_ctx.p              = mi.p;
                LightSample light_sample = light->sample_incidence(light_ctx);
                if (light_sample.pdf <= 0.0)
                    return astd::nullopt;
                light_pdf *= light_sample.pdf;
                auto phase = mi.phase.evaluate(std::abs(dot(vertex.wo, light_sample.wi)));
                AKR_CHECK(std::isfinite(phase));
                AKR_CHECK(std::isfinite(light_sample.pdf));
                AKR_CHECK(std::isfinite(light_pdf));
                AKR_CHECK(std::isfinite(hsum(light_sample.I)));
                lighting.radiance   = light_sample.I * phase / light_pdf * mis_weight(light_pdf, phase);
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf        = light_pdf;
                lighting.wi         = light_sample.wi;
                return lighting;
            }
            return astd::nullopt;
        }
        astd::optional<DirectLighting>
        compute_direct_lighting(const SurfaceVertex &vertex,
                                const std::pair<const Light<CPU> *, Float> &selected) noexcept {
            auto [light, light_pdf] = selected;
            if (light) {
                auto &si = vertex.si;
                DirectLighting lighting;
                LightSampleContext light_ctx;
                light_ctx.u              = sampler->next2d();
                light_ctx.p              = si.p;
                LightSample light_sample = light->sample_incidence(light_ctx);
                if (light_sample.pdf <= 0.0)
                    return astd::nullopt;
                light_pdf *= light_sample.pdf;
                auto f              = vertex.bsdf->evaluate(vertex.wo, light_sample.wi);
                Float bsdf_pdf      = vertex.bsdf->evaluate_pdf(vertex.wo, light_sample.wi);
                lighting.throughput = light_sample.I * std::abs(dot(si.ns, light_sample.wi)) / light_pdf *
                                      mis_weight(light_pdf, bsdf_pdf);
                lighting.radiance   = f * lighting.throughput;
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf        = light_pdf;
                lighting.wi         = light_sample.wi;
                if (visitor.on_direct_lighting(vertex, lighting)) {
                    return lighting;
                }
                return astd::nullopt;
            } else {
                return astd::nullopt;
            }
        }

        void on_miss(const Ray &ray, const astd::optional<PathVertex> &prev_vertex) noexcept {
            // if (scene->envmap) {
            //     on_hit_light(scene->envmap.get(), -ray.d, ShadingPoint(), prev_vertex);
            // }
        }

        void accumulate_radiance(const Spectrum &r) { L += r; }

        void on_hit_light(const Light<CPU> *light, const Vec3 &wo, const ShadingPoint &sp,
                          const astd::optional<PathVertex> &prev_vertex) {
            Spectrum I = beta * light->Le(wo, sp);
            if (!config.use_nee || !prev_vertex || depth == 0 ||
                BSDFType::Unset != (prev_vertex->sampled_lobe() & BSDFType::Specular)) {
                ;
            } else {
                PointGeometry ref;
                ref.n          = prev_vertex->ng();
                ref.p          = prev_vertex->p();
                auto light_pdf = light->pdf_incidence(ref, -wo) * scene->light_sampler->pdf(light);
                if ((prev_vertex->sampled_lobe() & BSDFType::Specular) == BSDFType::Unset) {
                    Float weight_bsdf = mis_weight(prev_vertex->pdf(), light_pdf);
                    I *= weight_bsdf;
                }
            }
            HitLight hit{wo, light, I};
            if (visitor.on_hit_light(prev_vertex, hit)) {
                accumulate_radiance(I);
            }
        }
        void accumulate_beta(const Spectrum &k) { beta *= k; }
        astd::optional<MediumVertex> on_volume_scatter(const Vec3 &wo, const MediumInteraction &mi) noexcept {
            if (depth < max_depth) {
                MediumVertex vertex(wo, mi);
                auto [wi, phase] = mi.phase.sample(wo, sampler->next2d());
                vertex.ray       = Ray(mi.p, wi, Eps);
                vertex.pdf       = phase;
                vertex.beta      = Spectrum(1.0);
                return vertex;
            }
            return astd::nullopt;
        }
        astd::optional<SurfaceVertex> on_surface_scatter(const Vec3 &wo, SurfaceInteraction<CPU> &si) noexcept {
            auto *material = si.material();
            if (depth < max_depth) {
                SurfaceVertex vertex(wo, si);
                auto bsdf = material->evaluate(*sampler, allocator, si);
                BSDFSampleContext sample_ctx{sampler->next1d(), sampler->next2d(), wo};
                auto sample = bsdf.sample(sample_ctx);
                if (!sample) {
                    return astd::nullopt;
                }
                AKR_ASSERT(sample->pdf >= 0.0f);
                if (sample->pdf == 0.0f) {
                    return astd::nullopt;
                }
                vertex.bsdf = bsdf;
                vertex.ray  = spawn_ray(si.p, sample->wi, si.ng);
                // vertex.ray = Ray(si.p, sample->wi, Eps / std::abs(glm::dot(si.ng, sample->wi)));
                vertex.beta         = sample->f * (std::abs(glm::dot(si.ns, sample->wi)) / sample->pdf);
                vertex.pdf          = sample->pdf;
                vertex.sampled_lobe = sample->type;
                return vertex;
            }
            return astd::nullopt;
        }
        Spectrum transmittance(Ray shadow_ray, MediumStack<CPU> st) {
            Spectrum tr(1.0);
            vec3 dst = shadow_ray(shadow_ray.tmax);
            int iter = 0;
            while (true) {
                Ray ray = shadow_ray;
                auto si = scene->intersect(ray);
                if (!st.empty()) {
                    tr *= st.top()->transmittance(ray, *sampler);
                }
                iter++;
                if (iter > 1024) {
                    fprintf(stderr, "Max iteration in transmittance() reached\n");
                    break;
                }
                if (is_black(tr) || hmax(tr) < 1e-10)
                    return Spectrum(0);
                if (si) {
                    if (si->material() != nullptr)
                        return Spectrum(0);
                    st.update(*si, ray);
                } else {
                    break;
                }
                shadow_ray.o    = offset_ray(si->p, dot(shadow_ray.d, si->ng) > 0 ? si->ng : -si->ng);
                shadow_ray.tmax = length(dst - si->p) / length(shadow_ray.d);
            }
            AKR_CHECK(!std::isnan(hsum(tr)));
            return tr;
        }
        void run_megakernel(Ray ray, MediumStack<CPU> st, astd::optional<PathVertex> prev_vertex) noexcept {
            astd::optional<PathVertex> this_vertex;
            while (true) {
                auto si = scene->intersect(ray);
                auto wo = -ray.d;
                // if (!si) {
                //     on_miss(ray, prev_vertex);
                //     break;
                // }
                astd::optional<MediumInteraction> mi;
                if (si) {
                    if (config.volumetric && !st.empty()) {
                        Spectrum tr(0);
                        auto _ = st.top()->sample(ray, *sampler, allocator);
                        mi     = _.first;
                        tr     = _.second;
                        AKR_CHECK(!std::isnan(hsum(tr)));
                        accumulate_beta(tr);
                    }
                    if (!mi)
                        st.update(*si, ray);

                    if (si->triangle.light) {
                        on_hit_light(si->triangle.light, wo, si->sp(), prev_vertex);
                        break;
                    }
                }
                if (!mi) {
                    if (!si) {
                        on_miss(ray, prev_vertex);
                        break;
                    }
                    if (!si->material() && si->medium()) {
                        // skip volume boundary
                        // ray = Ray(si->p, ray.d, Eps);
                        ray = spawn_ray(si->p, ray.d, si->ng);
                        continue;
                    }
                    auto vertex = on_surface_scatter(wo, *si);
                    if (!vertex || !visitor.on_scatter(*vertex, prev_vertex)) {
                        break;
                    }
                    if (config.use_nee && (vertex->sampled_lobe & BSDFType::Specular) == BSDFType::Unset) {
                        astd::optional<DirectLighting> has_direct = compute_direct_lighting(*vertex, select_light());
                        if (has_direct) {
                            auto &direct = *has_direct;
                            if (!is_black(direct.radiance())) {
                                if (!config.volumetric) {
                                    if (!scene->occlude(direct.shadow_ray))
                                        accumulate_radiance(beta * direct.radiance());
                                } else {
                                    accumulate_radiance(beta * direct.radiance() *
                                                        transmittance(direct.shadow_ray, st));
                                }
                            }
                        }
                    }
                    accumulate_beta(vertex->beta());
                    ray         = vertex->ray;
                    this_vertex = PathVertex(*vertex);
                } else {
                    auto vertex = on_volume_scatter(wo, *mi);
                    if (!vertex) {
                        break;
                    }
                    if (config.use_nee) {
                        astd::optional<VolumeDirectLighting> has_direct =
                            compute_direct_lighting(*vertex, select_light());
                        if (has_direct) {
                            auto &direct = *has_direct;
                            AKR_CHECK(!std::isnan(hsum(direct.radiance)));
                            if (!is_black(direct.radiance)) {
                                accumulate_radiance(beta * direct.radiance * transmittance(direct.shadow_ray, st));
                            }
                        }
                    }
                    AKR_CHECK(!std::isnan(hsum(vertex->beta)));
                    accumulate_beta(vertex->beta);
                    ray         = vertex->ray;
                    this_vertex = PathVertex(*vertex);
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

                prev_vertex = this_vertex;
            }
        }
        void run_megakernel(const Camera<CPU> *camera, const ivec2 &p) noexcept {
            auto camera_sample = camera_ray(camera, p);
            Ray ray            = camera_sample.ray;
            MediumStack<CPU> st;
            run_megakernel(ray, st, astd::nullopt);
        }
    };

} // namespace akari::render::pt

namespace akari::render {
    Spectrum pt_estimator(PTConfig config, const Scene<CPU> &scene, Allocator<> alloc, Sampler<CPU> &sampler, Ray &ray,
                          astd::optional<pt::PathVertex> prev_vertex) {
        pt::SimplePathTracer<> pt(&scene, &sampler, alloc, config.min_depth, config.max_depth);
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