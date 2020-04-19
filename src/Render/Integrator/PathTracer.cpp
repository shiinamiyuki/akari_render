// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#include <Akari/Core/Logger.h>
#include <Akari/Core/Parallel.h>
#include <Akari/Core/Plugin.h>
#include <Akari/Core/Profiler.hpp>
#include <Akari/Core/Progress.hpp>
#include <Akari/Render/Integrator.h>
#include <future>
#include <mutex>
namespace Akari {
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
        bool HasFilmUpdate() override { return false; }
        std::shared_ptr<const Film> GetFilmUpdate() override { return ctx.camera->GetFilm(); }
        bool IsDone() override { return false; }
        bool WaitEvent(Event event) override {
            if (event == Event::EFILM_AVAILABLE) {
                std::unique_lock<std::mutex> lock(mutex);

                filmAvailable.wait(lock, [=]() { return HasFilmUpdate() || IsDone(); });
                return HasFilmUpdate();
            } else if (event == Event::ERENDER_DONE) {
                std::unique_lock<std::mutex> lock(mutex);

                done.wait(lock, [=]() { return IsDone(); });
                return true;
            }
            return false;
        }
        void Wait() override {
            future.wait();
            done.notify_all();
        }
        static std::string PrintVec3(const vec3 &v) { return fmt::format("{} {} {}", v.x, v.y, v.z); }
        Spectrum Li(Ray ray, Sampler *sampler, MemoryArena &arena) {
            auto scene = ctx.scene;

            Spectrum Li(0), beta(1);
            bool specular = false;
            Float prevScatteringPdf = 0;
            Interaction *prevInteraction = nullptr;
            int depth = 0;
            while (true) {
                Intersection intersection(ray);
                if (scene->Intersect(ray, &intersection)) {
                    auto &mesh = scene->GetMesh(intersection.meshId);
                    int group = mesh.GetPrimitiveGroup(intersection.primId);
                    const auto &materialSlot = mesh.GetMaterialSlot(group);
                    const auto *light = mesh.GetLight(intersection.primId);

                    auto material = materialSlot.material;
                    if (!material) {
                        Debug("no material!!\n");
                        break;
                    }
                    Triangle triangle{};
                    mesh.GetTriangle(intersection.primId, &triangle);
                    const auto &p = intersection.p;
                    auto *si = arena.alloc<SurfaceInteraction>(&materialSlot, -ray.d, p, triangle, intersection, arena);
                    si->ComputeScatteringFunctions(arena, TransportMode::EImportance, 1.0f);
                    auto Le = si->Le(si->wo);
                    if (!Le.IsBlack()) {
                        if (!light || specular || depth == 0)
                            Li += beta * (light->Li(si->wo, si->uv));
                        else {
                            auto lightPdf = light->PdfIncidence(*prevInteraction, ray.d) * scene->PdfLight(light);
                            Li += beta * (light->Li(si->wo, si->uv) * MisWeight(prevScatteringPdf, lightPdf));
                        }
                    }
                    if (depth++ >= maxDepth) {
                        break;
                    }
                    BSDFSample bsdfSample(sampler->Next1D(), sampler->Next2D(), *si);
                    si->bsdf->Sample(bsdfSample);

                    assert(bsdfSample.pdf >= 0);
                    if (bsdfSample.pdf <= 0) {
                        break;
                    }
                    specular = bsdfSample.sampledType & BSDF_SPECULAR;
                    {
                        Float lightPdf = 0;
                        auto sampledLight = scene->SampleOneLight(sampler->Next1D(), &lightPdf);
                        if (sampledLight && lightPdf > 0) {
                            LightSample lightSample{};
                            VisibilityTester tester{};
                            sampledLight->SampleIncidence(sampler->Next2D(), *si, &lightSample, &tester);
                            lightPdf *= lightSample.pdf;
                            auto wi = lightSample.wi;
                            auto wo = si->wo;
                            auto f = si->bsdf->Evaluate(wo, wi) * abs(dot(lightSample.wi, si->Ns));
                            if (lightPdf > 0 && MaxComp(f) > 0 && tester.visible(*scene)) {
                                if (specular || Light::IsDelta(sampledLight->GetLightType())) {
                                    Li += beta * f * lightSample.I / lightPdf;
                                } else {
                                    auto scatteringPdf = si->bsdf->EvaluatePdf(wo, wi);
                                    Li += beta * f * lightSample.I / lightPdf * MisWeight(lightPdf, scatteringPdf);
                                }
                            }
                        }
                    }
                    prevScatteringPdf = bsdfSample.pdf;
                    auto wiW = bsdfSample.wi;
                    beta *= bsdfSample.f * abs(dot(wiW, si->Ns)) / bsdfSample.pdf;
                    if (enableRR && depth > minDepth) {
                        Float continueProb = std::min(0.95f, MaxComp(beta));
                        if (sampler->Next1D() < continueProb) {
                            beta /= continueProb;
                        } else {
                            break;
                        }
                    }
                    ray = si->SpawnRay(wiW);
                    prevInteraction = si;
                } else {
                    Li += beta * Spectrum(0);
                    break;
                }
            }
            return Li;
        }
        void Start() override {
            future = std::async(std::launch::async, [=]() {
                auto beginTime = std::chrono::high_resolution_clock::now();

                auto scene = ctx.scene;
                auto &camera = ctx.camera;
                auto &_sampler = ctx.sampler;
                auto film = camera->GetFilm();
                auto nTiles = ivec2(film->Dimension() + ivec2(TileSize - 1)) / ivec2(TileSize);
                ProgressReporter progressReporter(nTiles.x * nTiles.y, [=](size_t cur, size_t tot) {
                    if (spp <= 16) {
                        if (cur % (tot / 10) == 0) {
                            ShowProgress(double(cur) / tot, 70);
                        }
                    } else {
                        ShowProgress(double(cur) / tot, 70);
                    }
                });
                ParallelFor2D(nTiles, [=, &progressReporter](ivec2 tilePos, uint32_t tid) {
                    (void)tid;
                    MemoryArena arena;
                    Bounds2i tileBounds = Bounds2i{tilePos * (int)TileSize, (tilePos + ivec2(1)) * (int)TileSize};
                    auto tile = film->GetTile(tileBounds);
                    auto sampler = _sampler->Clone();
                    for (int y = tile.bounds.p_min.y; y < tile.bounds.p_max.y; y++) {
                        for (int x = tile.bounds.p_min.x; x < tile.bounds.p_max.x; x++) {
                            sampler->SetSampleIndex(x + y * film->Dimension().x);
                            for (int s = 0; s < spp; s++) {
                                sampler->StartNextSample();
                                CameraSample sample;
                                camera->GenerateRay(sampler->Next2D(), sampler->Next2D(), ivec2(x, y), &sample);
                                auto Li = this->Li(sample.primary, sampler.get(), arena);
                                arena.reset();
                                tile.AddSample(ivec2(x, y), Li, 1.0f);
                            }
                        }
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    film->MergeTile(tile);
                    progressReporter.Update();
                });
                auto endTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = (endTime - beginTime);
                Info("Rendering done in {} secs, traced {} rays, {} M rays/sec\n", elapsed.count(),
                     scene->GetRayCounter(), scene->GetRayCounter() / elapsed.count() / 1e6);
            });
        }
    };

    class PathTracer : public Integrator {
        int spp = 16;
        int minDepth = 5, maxDepth = 16;
        bool enableRR = true;

      public:
        AKR_DECL_COMP(PathTracer, "PathTracer")
        AKR_SER(spp, minDepth, maxDepth, enableRR)
        std::shared_ptr<RenderTask> CreateRenderTask(const RenderContext &ctx) override {
            return std::make_shared<PTRenderTask>(ctx, spp, minDepth, maxDepth, enableRR);
        }
    };
    AKR_EXPORT_COMP(PathTracer, "Integrator");
} // namespace Akari