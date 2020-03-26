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
#include <Akari/Render/Integrator.h>
#include <future>
#include <mutex>

namespace Akari {
    struct RTDebugRenderTask : RenderTask {
        RenderContext ctx;
        std::mutex mutex;
        std::condition_variable filmAvailable, done;
        std::future<void> future;
        int spp;
        RTDebugRenderTask(const RenderContext &ctx, int spp) : ctx(ctx), spp(spp) {}
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
            for (int depth = 0; depth < 5; depth++) {
                Intersection intersection(ray);
                if (scene->Intersect(ray, &intersection)) {
                    auto &mesh = scene->GetMesh(intersection.meshId);
                    int group = mesh.GetPrimitiveGroup(intersection.primId);
                    const auto &materialSlot = mesh.GetMaterialSlot(group);
                    auto material = materialSlot.material;
                    if (!material) {
                        Debug("no material!!\n");
                        break;
                    }
                    Triangle triangle{};
                    mesh.GetTriangle(intersection.primId, &triangle);
                    vec3 p = ray.At(intersection.t);
                    SurfaceInteraction si(-ray.d, p, triangle, intersection, arena);
                    material->computeScatteringFunctions(&si, arena, TransportMode::EImportance, 1.0f);
                    BSDFSample bsdfSample(sampler->Next1D(), sampler->Next2D(), si);
                    si.bsdf->Sample(bsdfSample);
                    //                                    Debug("pdf:{}\n",bsdfSample.pdf);
                    assert(bsdfSample.pdf >= 0);
                    if (bsdfSample.pdf <= 0) {
                        break;
                    }
                    //                                    Debug("wi: {}\n",PrintVec3(bsdfSample.wi));
                    auto wiW = si.bsdf->LocalToWorld(bsdfSample.wi);
                    beta *= bsdfSample.f * abs(dot(wiW, si.Ns)) / bsdfSample.pdf;
                    ray = si.SpawnRay(wiW);
                } else {
                    Li += beta * Spectrum(1);
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
                ParallelFor2D(nTiles, [=](ivec2 tilePos, uint32_t tid) {
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
                });
                auto endTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = (endTime - beginTime);
                Info("Rendering done in {} secs, traced {} rays, {} M rays/sec\n", elapsed.count(),
                     scene->GetRayCounter(), scene->GetRayCounter() / elapsed.count() / 1e6);
            });
        }
    };

    class RTDebug : public Integrator {
        int spp = 16;

      public:
        AKR_DECL_COMP(RTDebug, "RTDebug")
        AKR_SER(spp)
        std::shared_ptr<RenderTask> CreateRenderTask(const RenderContext &ctx) override {
            return std::make_shared<RTDebugRenderTask>(ctx, spp);
        }
    };
    AKR_EXPORT_COMP(RTDebug, "Integrator");
} // namespace Akari
