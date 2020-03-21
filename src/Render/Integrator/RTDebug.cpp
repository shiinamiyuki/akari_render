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
        explicit RTDebugRenderTask(const RenderContext &ctx) : ctx(ctx) {}
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
        void Start() override {
            future = std::async(std::launch::async, [=]() {
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
                            CameraSample sample;

                            sampler->SetSampleIndex(x + y * film->Dimension().x);
                            sampler->StartNextSample();
                            camera->GenerateRay(sampler->Next2D(), sampler->Next2D(), ivec2(x, y), sample);

                            auto ray = sample.primary;
                            Spectrum Li(0), beta(1);
                            for (int depth = 0; depth < 5; depth++) {
                                Intersection intersection;
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
                                    ScatteringEvent event(-ray.d, p, triangle, intersection);
                                    material->computeScatteringFunctions(&event, arena);
                                    BSDFSample bsdfSample(sampler->Next1D(), sampler->Next2D(), event);
                                    event.bsdf->Sample(bsdfSample);
                                    //                                    Debug("pdf:{}\n",bsdfSample.pdf);
                                    assert(bsdfSample.pdf >= 0);
                                    if (bsdfSample.pdf <= 0) {
                                        break;
                                    }
//                                    Debug("wi: {}\n",PrintVec3(bsdfSample.wi));
                                    auto wiW = event.bsdf->LocalToWorld(bsdfSample.wi);
                                    beta *= bsdfSample.f * abs(dot(wiW, event.Ns)) / bsdfSample.pdf;
                                    ray = event.SpawnRay(wiW);
                                } else {
                                    Li += beta * Spectrum(1);
                                    break;
                                }
                            }
                            arena.reset();
                            tile.AddSample(ivec2(x, y), Li, 1.0f);
                        }
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    film->MergeTile(tile);
                });
            });
        }
    };
    class RTDebug : public Integrator {
      public:
        AKR_DECL_COMP(RTDebug, "RTDebug")
        std::shared_ptr<RenderTask> CreateRenderTask(const RenderContext &ctx) override {
            return std::make_shared<RTDebugRenderTask>(ctx);
        }
    };
    AKR_EXPORT_COMP(RTDebug, "Integrator");
} // namespace Akari
