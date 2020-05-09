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

#include <akari/core/logger.h>
#include <akari/core/parallel.h>
#include <akari/core/plugin.h>
#include <akari/render/integrator.h>
#include <akari/plugins/core_bidir.h>
#include <future>
#include <mutex>
#include <utility>

namespace akari {

    class BDPTRenderTask : public RenderTask {
        RenderContext ctx;
        std::mutex mutex;
        std::condition_variable filmAvailable, done;
        std::future<void> future;
        int spp;
        int maxDepth;
        std::vector<std::shared_ptr<Film>> pyramid;
        inline int BufferIndex(int s, int t) { return s * (maxDepth + 2) + t; }
        bool visualizeMIS = false;

      public:
        BDPTRenderTask(RenderContext ctx, int spp, int maxDepth, bool visualizeMIS)
            : ctx(std::move(ctx)), spp(spp), maxDepth(maxDepth), visualizeMIS(visualizeMIS) {
            if (visualizeMIS) {
                pyramid.resize((maxDepth + 2) * (maxDepth + 2));
            }
        }
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
            size_t nCamera = trace_eye_path(scene, arena, camera, raster, *sampler, eyePath, maxDepth);
            size_t nLight = trace_light_path(scene, arena, *sampler, lightPath, maxDepth);
            Spectrum L(0);
            for (size_t t = 1; t <= nCamera; ++t) {
                for (size_t s = 0; s <= nLight; ++s) {
                    int depth = int(t + s) - 2;
                    if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth)
                        continue;
                    vec2 pRaster = raster;
                    Spectrum LPath = connect_path(scene, *sampler, eyePath, t, lightPath, s, &pRaster);
                    if (t != 1) {
                        if (visualizeMIS) {
                            pyramid.at(BufferIndex(s, t))->add_splat(LPath, raster);
                        }
                        L += LPath;
                    } else {
                        if (visualizeMIS) {
                            pyramid.at(BufferIndex(s, t))->add_splat(LPath, pRaster);
                        }
                        film->add_splat(LPath, pRaster);
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
                if (visualizeMIS) {
                    for (auto &p : pyramid) {
                        p = std::make_shared<Film>(film->Dimension());
                    }
                }
                parallel_for_2d(nTiles, [=](ivec2 tilePos, uint32_t tid) {
                    (void) tid;
                    MemoryArena arena;
                    Bounds2i tileBounds = Bounds2i{tilePos * (int) TileSize, (tilePos + ivec2(1)) * (int) TileSize};
                    auto tile = film->GetTile(tileBounds);
                    auto sampler = _sampler->clone();
                    for (int y = tile.bounds.p_min.y; y < tile.bounds.p_max.y; y++) {
                        for (int x = tile.bounds.p_min.x; x < tile.bounds.p_max.x; x++) {
                            sampler->set_sample_index(x + y * film->Dimension().x);
                            for (int s = 0; s < spp; s++) {
                                sampler->start_next_sample();
                                auto Li = this->Li(film.get(), *scene, *camera, ivec2(x, y), sampler.get(), arena);
                                arena.reset();
                                tile.AddSample(ivec2(x, y), Li, 1.0f);
                            }
                        }
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    film->merge_tile(tile);
                });
                auto endTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = (endTime - beginTime);
                Info("Rendering done in {} secs, traced {} rays, {} M rays/sec\n", elapsed.count(),
                     scene->GetRayCounter(), scene->GetRayCounter() / elapsed.count() / 1e6);
                if (visualizeMIS) {
                    for (int s = 0; s <= maxDepth; s++) {
                        for (int t = 1; t <= maxDepth; t++) {
                            int depth = int(t + s) - 2;
                            if ((s == 1 && t == 1) || depth < 0 || depth > maxDepth)
                                continue;
                            auto &p = pyramid.at(BufferIndex(s, t));
                            p->splatScale = 1.0 / spp;
                            p->write_image(fmt::format("bdpt_d{}_s{}_t{}.png", s + t - 2, s, t));
                        }
                    }
                }
            });
        }
    };
    class BDPT : public Integrator {
        int spp = 4;
        int maxDepth = 5;
        bool visualize_mis = false;

      public:
        AKR_DECL_COMP()
        AKR_SER(spp, maxDepth, visualize_mis)
        std::shared_ptr<RenderTask> create_render_task(const RenderContext &ctx) override {
            return std::make_shared<BDPTRenderTask>(ctx, spp, maxDepth, visualize_mis);
        }
    };
    AKR_EXPORT_PLUGIN(BDPT, p){
        auto c = class_<BDPT, Integrator, Component>("BDPT");
        c.constructor<>();
        c.method("save", &BDPT::save);
        c.method("load", &BDPT::load);
        c.method("create_render_task", &BDPT::create_render_task);
    }
} // namespace akari
