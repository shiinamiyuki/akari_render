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

#include <mutex>
#include <akari/core/parallel.h>
#include <akari/core/progress.hpp>
#include <akari/core/profiler.h>
#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/common.h>
#include <akari/render/pathtracer.h>

namespace akari::render {

    class AOVPathTracer {
      public:
        const Scene *scene = nullptr;
        Sampler *sampler = nullptr;
        Spectrum L;
        Spectrum beta = Spectrum(1.0f);
        struct DenoisingData {
            std::optional<Vec3> normal;
            std::optional<Spectrum> albedo;
            Vec3 first_hit_normal;
            Spectrum first_hit_albedo;
        };
        DenoisingData denoising;
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
                LightSample light_sample = light->sample_incidence(light_ctx);
                if (light_sample.pdf <= 0.0)
                    return std::nullopt;
                light_pdf *= light_sample.pdf;
                auto f = light_sample.I * si.bsdf.evaluate(surface_hit.wo, light_sample.wi) *
                         std::abs(dot(si.ns, light_sample.wi));
                Float bsdf_pdf = si.bsdf.evaluate_pdf(surface_hit.wo, light_sample.wi);
                lighting.color = f / light_pdf * mis_weight(light_pdf, bsdf_pdf);
                lighting.shadow_ray = light_sample.shadow_ray;
                lighting.pdf = light_pdf;
                return lighting;
            } else {
                return std::nullopt;
            }
        }

        void on_miss(const Ray &ray, const std::optional<PathVertex> &prev_vertex) noexcept {
            if (scene->envmap) {
                auto I = on_hit_light(scene->envmap.get(), -ray.d, ShadingPoint(), prev_vertex);
                if (!denoising.albedo) {
                    denoising.albedo = I;
                }
            }
        }

        void accumulate_radiance(const Spectrum &r) { L += r; }

        Spectrum on_hit_light(const Light *light, const Vec3 &wo, const ShadingPoint &sp,
                              const std::optional<PathVertex> &prev_vertex) {
            Spectrum I = beta * light->Le(wo, sp);
            if (depth == 0 || BSDFType::Unset != (prev_vertex->sampled_lobe() & BSDFType::Specular)) {
                accumulate_radiance(I);
                return I;
            } else {
                ReferencePoint ref;
                ref.ng = prev_vertex->ng();
                ref.p = prev_vertex->p();
                auto light_pdf = light->pdf_incidence(ref, -wo) * scene->pdf_light(light);
                if ((prev_vertex->sampled_lobe() & BSDFType::Specular) != BSDFType::Unset) {
                    accumulate_radiance(I);
                    return I;
                } else {
                    Float weight_bsdf = mis_weight(prev_vertex->pdf(), light_pdf);
                    accumulate_radiance(weight_bsdf * I);
                    return weight_bsdf * I;
                }
            }
        }
        void accumulate_beta(const Spectrum &k) { beta *= k; }
        // @param mat_pdf: supplied if material is already chosen
        std::optional<SurfaceVertex> on_surface_scatter(SurfaceInteraction &si, const SurfaceHit &surface_hit,
                                                        const std::optional<PathVertex> &prev_vertex) noexcept {
            auto *material = surface_hit.material;
            auto &wo = surface_hit.wo;
            MaterialEvalContext ctx = si.mat_eval_ctx(allocator, sampler);
            if (depth == 0) {
                denoising.first_hit_albedo = material->albedo(ctx.sp);
                denoising.first_hit_normal = si.ns;
            }
            {
                auto u = sampler->next1d();
                auto roughness = material->roughness(ctx.sp);
                if (u < roughness) {
                    if (!denoising.normal) {
                        denoising.normal = si.ns;
                    }
                    if (!denoising.albedo) {
                        denoising.albedo = material->albedo(ctx.sp);
                    }
                }
            }
            if (si.triangle.light) {
                auto I = on_hit_light(si.triangle.light, wo, ctx.sp, prev_vertex);
                if (!denoising.albedo) {
                    denoising.albedo = I;
                }
                if (!denoising.normal) {
                    denoising.normal = si.ns;
                }
                return std::nullopt;
            } else if (depth < max_depth) {
                SurfaceVertex vertex(si.triangle, surface_hit);
                si.bsdf = material->get_bsdf(ctx);

                BSDFSampleContext sample_ctx(sampler->next2d(), wo);
                auto sample = si.bsdf.sample(sample_ctx);
                if (!sample) {
                    return std::nullopt;
                }
                AKR_ASSERT(sample->pdf >= 0.0f);
                if (sample->pdf == 0.0f) {
                    return std::nullopt;
                }
                vertex.bsdf = si.bsdf;
                vertex.ray = Ray(si.p, sample->wi, Eps / std::abs(glm::dot(si.ng, sample->wi)));
                vertex.beta = sample->f * std::abs(glm::dot(si.ns, sample->wi)) / sample->pdf;
                vertex.pdf = sample->pdf;
                return vertex;
            }
            return std::nullopt;
        }
        void run_megakernel(const Camera *camera, const ivec2 &p) noexcept {
            auto camera_sample = camera_ray(camera, p);
            Ray ray = camera_sample.ray;
            std::optional<PathVertex> prev_vertex;
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

                if ((vertex->sampled_lobe & BSDFType::Specular) == BSDFType::Unset) {
                    std::optional<DirectLighting> has_direct = compute_direct_lighting(si, surface_hit, select_light());
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
    template <class T>
    struct VarianceTracker {
        std::optional<T> mean, m2;
        int count = 0;
        void update(T value) {
            if (count == 0) {
                mean = value;
                m2 = T(0.0);
            } else {
                auto delta = value - *mean;
                *mean += delta / T(count + 1);
                *m2 += delta * (value - *mean);
            }
            count++;
        }
        std::optional<T> variance() const {
            if (count < 2) {
                return std::nullopt;
            }
            return *m2 / float(count * count);
        }
    };
    template <class T>
    struct Statistics {
        using value_type = T;
        T value = T(0.0);
        VarianceTracker<T> var;
        std::pair<VarianceTracker<T>, VarianceTracker<T>> half_vars;
        std::pair<T, T> halves = std::make_pair(T(0.0), T(0.0));
        std::pair<int, int> counts = std::make_pair(0, 0);
        std::pair<T, T> compute_halves() const {
            return std::make_pair(halves.first / float(counts.first), halves.second / float(counts.second));
        }
        T compute_value() const { return value / float(count); }
        int total_samples = 0;
        int count = 0;
        void set_total_samples(int N) { total_samples = N; }
        void add_sample(const T v) {
            value += v;
            var.update(v);
            if (count < total_samples / 2) {
                halves.first += v;
                counts.first++;
                half_vars.first.update(v);
            } else {
                halves.second += v;
                counts.second++;
                half_vars.second.update(v);
            }
            count++;
        }
    };
    using VarianceTracker1f = VarianceTracker<double>;
    using VarianceTracker3f = VarianceTracker<Vec3>;
    class PathTracerIntegrator : public Integrator {
        int spp;
        int min_depth;
        int max_depth;
        const int tile_size = 16;
        Float ray_clamp;

      public:
        PathTracerIntegrator(int spp, int min_depth, int max_depth, Float ray_clamp)
            : spp(spp), min_depth(min_depth), max_depth(max_depth), ray_clamp(ray_clamp) {}
        RenderOutput render(const RenderInput &) override;
    };
    class PathIntegratorNode final : public IntegratorNode {
      public:
        int spp = 16;
        int max_depth = 5;
        int min_depth = 3;
        Float ray_clamp = 10;
        std::shared_ptr<Integrator> create_integrator(Allocator<> allocator) override {
            return make_pmr_shared<PathTracerIntegrator>(allocator, spp, min_depth, max_depth, ray_clamp);
        }
        const char *description() override { return "[Path Tracer]"; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "spp") {
                spp = value.get<int>().value();
            } else if (field == "max_depth") {
                max_depth = value.get<int>().value();
            } else if (field == "min_depth") {
                min_depth = value.get<int>().value();
            } else if (field == "clamp") {
                ray_clamp = value.get<float>().value();
            }
        }
        bool set_spp(int spp_) override {
            spp = spp_;
            return true;
        }
        int get_spp() const override { return spp; }
    };
    AKR_EXPORT_NODE(Path, PathIntegratorNode)
} // namespace akari::render

akari::render::RenderOutput akari::render::PathTracerIntegrator::render(const akari::render::RenderInput &input) {
    auto scene = input.scene;
    bool uni_aov = input.required_full_aov == false && input.requested_aovs.size() == 0;
    RenderOutput out;
    const auto resolution = scene->camera->resolution();
    for (auto &aov : input.requested_aovs) {
        out.aovs[aov.first].value = Film(resolution);
    }
    bool require_full_aov = input.required_full_aov;
    auto setup_aov1 = [&](const std::string &name) {
        auto &rec = out.aovs[name];
        if (!rec.value)
            rec.value = Film(resolution);
        if (require_full_aov)
            rec.variance = Film(resolution);
    };
    auto setup_aov = [&](const std::string &name) {
        setup_aov1(name);
        if (require_full_aov) {
            setup_aov1(name + "_A");
            setup_aov1(name + "_B");
        }
    };
    setup_aov("color");
    setup_aov("normal");
    setup_aov("albedo");

    info("Path Tracer");
    const auto n_tiles = ivec2(resolution + ivec2(tile_size - 1)) / ivec2(tile_size);
    debug("resolution: {}, tile size: {}, tiles: {}", resolution, tile_size, n_tiles);
    std::mutex mutex;
    std::vector<astd::pmr::monotonic_buffer_resource *> resources;
    for (size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
        resources.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::get_default_resource()));
    }
    Timer timer;
    int estimate_ray_per_sample = max_depth * 2 + 1;
    double estimate_ray_per_sec = 5 * 1000 * 1000;
    double estimate_single_tile = spp * estimate_ray_per_sample * tile_size * tile_size / estimate_ray_per_sec;
    size_t estimate_tiles_per_sec = std::max<size_t>(1, size_t(1.0 / estimate_single_tile));
    debug("estimate_tiles_per_sec:{} total:{}", estimate_tiles_per_sec, n_tiles.x * n_tiles.y);
    auto reporter = std::make_shared<ProgressReporter>(n_tiles.x * n_tiles.y, [=, &mutex](size_t cur, size_t total) {
        std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);

        bool show = (0 == cur % (estimate_tiles_per_sec));
        if (!lock.owns_lock())
            show = false;
        if (show) {
            double tiles_per_sec = cur / std::max(1e-7, timer.elapsed_seconds());
            double remaining = (total - cur) / tiles_per_sec;
            show_progress(double(cur) / double(total), timer.elapsed_seconds(), remaining);
        }
        if (cur == total) {
            putchar('\n');
        }
    });
    bool require_albedo = input.requested_aovs.find("albedo") != input.requested_aovs.end();
    bool require_normal = input.requested_aovs.find("normal") != input.requested_aovs.end();
    bool require_first_hit_albedo = input.requested_aovs.find("first_hit_albedo") != input.requested_aovs.end();
    bool require_first_hit_normal = input.requested_aovs.find("first_hit_normal") != input.requested_aovs.end();
    const Bounds2i film_bounds(ivec2(0), resolution);
    thread::parallel_for(thread::blocked_range<2>(n_tiles), [&](const ivec2 &tile_pos, int tid) {
        Allocator<> allocator(resources[tid]);
        Bounds2i tile_bounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + ivec2(1)) * (int)tile_size};
        tile_bounds = tile_bounds.intersect(film_bounds);
        auto camera = scene->camera;
        auto sampler = scene->sampler->clone(Allocator<>());
        for (int y = tile_bounds.pmin.y; y < tile_bounds.pmax.y; y++) {
            for (int x = tile_bounds.pmin.x; x < tile_bounds.pmax.x; x++) {
                sampler->set_sample_index(x + y * resolution.x);
                auto p_raster = ivec2(x, y);
                if (uni_aov) {
                    Spectrum L(0.0);
                    for (int s = 0; s < spp; s++) {
                        sampler->start_next_sample();
                        GenericPathTracer pt;
                        pt.scene = scene;
                        pt.allocator = allocator;
                        pt.sampler = sampler.get();
                        pt.L = Spectrum(0.0);
                        pt.beta = Spectrum(1.0);
                        pt.min_depth = min_depth;
                        pt.max_depth = max_depth;
                        pt.run_megakernel(camera.get(), p_raster);
                        L += min(clamp_zero(pt.L), Spectrum(ray_clamp));
                    }
                    out.aovs.at("color").value->add_sample(p_raster, L / Float(spp), 1.0f);
                } else {
                    Statistics<Vec3> normal, first_hit_normal;
                    Statistics<Spectrum> color, albedo, first_hit_albedo;
                    {
                        normal.set_total_samples(spp);
                        first_hit_normal.set_total_samples(spp);
                        first_hit_albedo.set_total_samples(spp);
                        color.set_total_samples(spp);
                        albedo.set_total_samples(spp);
                    }
                    for (int s = 0; s < spp; s++) {
                        sampler->start_next_sample();
                        AOVPathTracer pt;
                        pt.scene = scene;
                        pt.allocator = allocator;
                        pt.sampler = sampler.get();
                        pt.L = Spectrum(0.0);
                        pt.beta = Spectrum(1.0);
                        pt.min_depth = min_depth;
                        pt.max_depth = max_depth;
                        pt.run_megakernel(camera.get(), p_raster);

                        auto add_sample = [&](auto &data, auto &stat) {
                            using T = typename std::decay_t<decltype(stat)>::value_type;
                            if (data) {
                                stat.add_sample(*data);
                            } else {
                                stat.add_sample(T(0.0));
                            }
                        };

                        add_sample(pt.denoising.albedo, albedo);
                        add_sample(pt.denoising.normal, normal);
                        first_hit_normal.add_sample(pt.denoising.first_hit_normal);
                        first_hit_albedo.add_sample(pt.denoising.first_hit_albedo);
                        color.add_sample(min(clamp_zero(pt.L), Spectrum(ray_clamp)));
                        resources[tid]->release();
                    }
                    out.aovs.at("color").value->add_sample(p_raster, color.compute_value(), 1.0f);
                    if (require_albedo) {
                        out.aovs.at("albedo").value->add_sample(p_raster, albedo.compute_value(), 1.0f);
                    }
                    if (require_normal) {
                        out.aovs.at("normal").value->add_sample(p_raster, normal.compute_value(), 1.0f);
                    }
                    if (require_first_hit_albedo) {
                        out.aovs.at("first_hit_albedo")
                            .value->add_sample(p_raster, first_hit_albedo.compute_value(), 1.0f);
                    }
                    if (require_first_hit_normal) {
                        out.aovs.at("first_hit_normal")
                            .value->add_sample(p_raster, first_hit_normal.compute_value(), 1.0f);
                    }
                    if (require_full_aov && spp >= 2) {
                        out.aovs.at("albedo").variance->add_sample(p_raster, *albedo.var.variance(), 1.0f);
                        out.aovs.at("albedo_A").value->add_sample(p_raster, albedo.compute_halves().first, 1.0f);
                        out.aovs.at("albedo_A")
                            .variance->add_sample(p_raster, *albedo.half_vars.first.variance(), 1.0f);
                        out.aovs.at("albedo_B").value->add_sample(p_raster, albedo.compute_halves().second, 1.0f);
                        out.aovs.at("albedo_B")
                            .variance->add_sample(p_raster, *albedo.half_vars.second.variance(), 1.0f);

                        out.aovs.at("normal").variance->add_sample(p_raster, *normal.var.variance(), 1.0f);
                        out.aovs.at("normal_A").value->add_sample(p_raster, normal.compute_halves().first, 1.0f);
                        out.aovs.at("normal_A")
                            .variance->add_sample(p_raster, *normal.half_vars.first.variance(), 1.0f);
                        out.aovs.at("normal_B").value->add_sample(p_raster, normal.compute_halves().second, 1.0f);
                        out.aovs.at("normal_B")
                            .variance->add_sample(p_raster, *normal.half_vars.second.variance(), 1.0f);

                        out.aovs.at("color").variance->add_sample(p_raster, *color.var.variance(), 1.0f);
                        out.aovs.at("color_A").value->add_sample(p_raster, color.compute_halves().first, 1.0f);
                        out.aovs.at("color_A").variance->add_sample(p_raster, *color.half_vars.first.variance(), 1.0f);
                        out.aovs.at("color_B").value->add_sample(p_raster, color.compute_halves().second, 1.0f);
                        out.aovs.at("color_B").variance->add_sample(p_raster, *color.half_vars.second.variance(), 1.0f);
                    }
                }
            }
        }
        reporter->update();
    });
    for (auto rsrc : resources) {
        delete rsrc;
    }
    return out;
}
