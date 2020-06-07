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

#include <akari/core/logger.h>
#include <akari/core/parallel.h>
#include <akari/core/plugin.h>
#include <akari/core/profiler.hpp>
#include <akari/core/progress.hpp>
#include <akari/render/integrator.h>
#include <future>
#include <mutex>
namespace akari {
    static Float MisWeight(Float pdfA, Float pdfB) {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    struct PTRenderTask : RenderTask {
        RenderContext ctx;
        std::mutex mutex;
        std::condition_variable filmAvailable, done;
        std::future<void> future;
        int spp;
        int minDepth;
        int maxDepth;
        bool enableRR;
        PTRenderTask(const RenderContext &ctx, int spp, int minDepth, int maxDepth, bool enableRR)
            : ctx(ctx), spp(spp), minDepth(minDepth), maxDepth(maxDepth), enableRR(enableRR) {}
        bool has_film_update() override { return false; }
        std::shared_ptr<const Film> film_update() override { return ctx.camera->GetFilm(); }
        bool is_done() override { return false; }
        bool wait_event(Event event) override {
            if (event == Event::EFILM_AVAILABLE) {
                std::unique_lock<std::mutex> lock(mutex);

                filmAvailable.wait(lock, [=]() { return has_film_update() || is_done(); });
                return has_film_update();
            } else if (event == Event::ERENDER_DONE) {
                std::unique_lock<std::mutex> lock(mutex);

                done.wait(lock, [=]() { return is_done(); });
                return true;
            }
            return false;
        }
        void wait() override {
            future.wait();
            done.notify_all();
        }
        static std::string PrintVec3(const vec3 &v) { return fmt::format("{} {} {}", v.x, v.y, v.z); }
        Spectrum Li(Ray ray, Sampler *sampler, MemoryArena &arena) {
            auto &scene = ctx.scene;

            Spectrum Li(0), beta(1);
            bool specular = false;
            Float prevScatteringPdf = 0;
            Interaction *prevInteraction = nullptr;
            int depth = 0;
            while (true) {
                Intersection intersection(ray);
                if (scene->intersect(ray, &intersection)) {
                    auto &mesh = scene->get_mesh(intersection.meshId);
                    int group = mesh.get_primitive_group(intersection.primId);
                    const auto &materialSlot = mesh.get_material_slot(group);
                    const auto *light = mesh.get_light(intersection.primId);

                    auto material = materialSlot.material;
                    if (!material) {
                        debug("no material!!\n");
                        break;
                    }
                    Triangle triangle{};
                    mesh.get_triangle(intersection.primId, &triangle);
                    const auto &p = intersection.p;
                    auto *si = arena.alloc<SurfaceInteraction>(&materialSlot, -ray.d, p, triangle, intersection, arena);
                    si->compute_scattering_functions(arena, TransportMode::EImportance, 1.0f);
                    auto Le = si->Le(si->wo);
                    if (!Le.is_black()) {
                        if (!light || specular || depth == 0)
                            Li += beta * (light->Li(si->wo, si->uv));
                        else {
                            auto lightPdf = light->pdf_incidence(*prevInteraction, ray.d) * scene->PdfLight(light);
                            Li += beta * (light->Li(si->wo, si->uv) * MisWeight(prevScatteringPdf, lightPdf));
                        }
                    }
                    if (depth++ >= maxDepth) {
                        break;
                    }
                    BSDFSample bsdfSample(sampler->next1d(), sampler->next2d(), *si);
                    si->bsdf->sample(bsdfSample);

                    assert(bsdfSample.pdf >= 0);
                    if (bsdfSample.pdf <= 0) {
                        break;
                    }
                    specular = bsdfSample.sampledType & BSDF_SPECULAR;
                    {
                        Float lightPdf = 0;
                        auto sampledLight = scene->sample_one_light(sampler->next1d(), &lightPdf);
                        if (sampledLight && lightPdf > 0) {
                            LightSample lightSample{};
                            VisibilityTester tester{};
                            sampledLight->sample_incidence(sampler->next2d(), *si, &lightSample, &tester);
                            lightPdf *= lightSample.pdf;
                            auto wi = lightSample.wi;
                            auto wo = si->wo;
                            auto f = si->bsdf->evaluate(wo, wi) * abs(dot(lightSample.wi, si->Ns));
                            if (lightPdf > 0 && max_comp(f) > 0 && tester.visible(*scene)) {
                                if (specular || Light::is_delta(sampledLight->get_light_type())) {
                                    Li += beta * f * lightSample.I / lightPdf;
                                } else {
                                    auto scatteringPdf = si->bsdf->evaluate_pdf(wo, wi);
                                    Li += beta * f * lightSample.I / lightPdf * MisWeight(lightPdf, scatteringPdf);
                                }
                            }
                        }
                    }
                    prevScatteringPdf = bsdfSample.pdf;
                    auto wiW = bsdfSample.wi;
                    beta *= bsdfSample.f * abs(dot(wiW, si->Ns)) / bsdfSample.pdf;
                    if (enableRR && depth > minDepth) {
                        Float continueProb = std::min(0.95f, max_comp(beta));
                        if (sampler->next1d() < continueProb) {
                            beta /= continueProb;
                        } else {
                            break;
                        }
                    }
                    ray = si->spawn_dir(wiW);
                    prevInteraction = si;
                } else {
                    Li += beta * Spectrum(0);
                    break;
                }
            }
            return Li;
        }
        void start() override {
            future = std::async(std::launch::async, [=]() {
                auto beginTime = std::chrono::high_resolution_clock::now();

                auto scene = ctx.scene;
                auto &camera = ctx.camera;
                auto &_sampler = ctx.sampler;
                auto film = camera->GetFilm();
                auto nTiles = ivec2(film->resolution() + ivec2(TileSize - 1)) / ivec2(TileSize);
                ProgressReporter progressReporter(nTiles.x * nTiles.y, [=](size_t cur, size_t tot) {
                    if (spp <= 16) {
                        if (cur % (tot / 10) == 0) {
                            show_progress(double(cur) / tot, 70);
                        }
                    } else if (spp <= 40) {
                        if (cur % (tot / 100) == 0) {
                            show_progress(double(cur) / tot, 70);
                        }

                    } else {
                        if (cur % (tot / 200) == 0) {
                            show_progress(double(cur) / tot, 70);
                        }
                    }
                });
                parallel_for_2d(nTiles, [=, &progressReporter](ivec2 tilePos, uint32_t tid) {
                    (void)tid;
                    MemoryArena arena;
                    Bounds2i tileBounds = Bounds2i{tilePos * (int)TileSize, (tilePos + ivec2(1)) * (int)TileSize};
                    auto tile = film->tile(tileBounds);
                    auto sampler = _sampler->clone();
                    for (int y = tile.bounds.p_min.y; y < tile.bounds.p_max.y; y++) {
                        for (int x = tile.bounds.p_min.x; x < tile.bounds.p_max.x; x++) {
                            sampler->set_sample_index(x + y * film->resolution().x);
                            for (int s = 0; s < spp; s++) {
                                sampler->start_next_sample();
                                CameraSample sample;
                                camera->generate_ray(sampler->next2d(), sampler->next2d(), ivec2(x, y), &sample);
                                auto Li = this->Li(sample.primary, sampler.get(), arena);
                                arena.reset();
                                tile.add_sample(ivec2(x, y), Li, 1.0f);
                            }
                        }
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    film->merge_tile(tile);
                    progressReporter.update();
                });
                auto endTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = (endTime - beginTime);
                info("Rendering done in {} secs, traced {} rays, {} M rays/sec\n", elapsed.count(),
                     scene->GetRayCounter(), scene->GetRayCounter() / elapsed.count() / 1e6);
            });
        }
    };

    class PathTracer : public Integrator {
        [[refl]] int spp = 16;
        [[refl]] int min_depth = 5;
        [[refl]] int max_depth = 16;
        [[refl]] bool enable_rr = true;

      public:
        AKR_IMPLS(Integrator)
        bool supports_mode(RenderMode mode) const { return true; }
        std::shared_ptr<RenderTask> create_render_task(const RenderContext &ctx) override {
            return std::make_shared<PTRenderTask>(ctx, spp, min_depth, max_depth, enable_rr);
        }
    };
#include "generated/PathTracer.hpp"
    AKR_EXPORT_PLUGIN(p) {}
} // namespace akari