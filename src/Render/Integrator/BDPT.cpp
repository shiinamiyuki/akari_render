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
#include <Akari/Render/Plugins/CoreBidir.h>
#include <future>
#include <mutex>

namespace Akari {
    class BDPTRenderTask : public RenderTask {
        RenderContext ctx;
        std::mutex mutex;
        std::condition_variable filmAvailable, done;
        std::future<void> future;
        int spp;
        int maxDepth;

      public:
        BDPTRenderTask(const RenderContext &ctx, int spp, int maxDepth) : ctx(ctx), spp(spp), maxDepth(maxDepth) {}
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
        Spectrum Li(Film *film, const Scene &scene, const Camera &camera, const vec2 &raster, Sampler *sampler,
                    MemoryArena &arena) {
            auto lightPath = arena.allocN<PathVertex>(maxDepth + 1);
            auto eyePath = arena.allocN<PathVertex>(maxDepth + 1);
            size_t nCamera = TraceEyePath(scene, arena, camera, raster, *sampler, eyePath, maxDepth);
            size_t nLight = TraceLightPath(scene, arena, *sampler, lightPath, maxDepth);
            Spectrum L(0);
            for (size_t t = 1; t <= nCamera; ++t) {
                for (size_t s = 0; s <= nLight; ++s) {
                    int depth = int(t + s) - 2;
                    if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth)
                        continue;
                    vec2 pRaster = raster;
                    Spectrum LPath = ConnectPath(scene, *sampler, eyePath, t, lightPath, s, &pRaster);
                    if (t != 1) {
                        L += LPath;
                    } else {
                        film->AddSplat(LPath, pRaster);
                    }
                }
            }
            return L;
        }
        void Start() override {
            future = std::async(std::launch::async, [=]() {
                auto beginTime = std::chrono::high_resolution_clock::now();

                auto &scene = ctx.scene;
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
                                auto Li = this->Li(film.get(), *scene, *camera, ivec2(x, y), sampler.get(), arena);
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
    class BDPT : public Integrator {
        int spp = 4;
        int maxDepth = 5;

      public:
        AKR_DECL_COMP(BDPT, "BDPT")
        AKR_SER(spp, maxDepth)
        std::shared_ptr<RenderTask> CreateRenderTask(const RenderContext &ctx) override {
            return std::make_shared<BDPTRenderTask>(ctx, spp, maxDepth);
        }
    };
    AKR_EXPORT_COMP(BDPT, "Integrator")
} // namespace Akari
