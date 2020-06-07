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

    struct AORenderTask : RenderTask {
        RenderContext ctx;
        std::mutex mutex;
        std::condition_variable filmAvailable, done;
        std::future<void> future;
        int spp;
        Float occlude_distance;
        AORenderTask(const RenderContext &ctx, int spp, Float occlude_distance)
            : ctx(ctx), spp(spp), occlude_distance(occlude_distance) {}
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

            Intersection intersection(ray);
            if (scene->intersect(ray, &intersection)) {
                auto &mesh = scene->get_mesh(intersection.meshId);

                Triangle triangle{};
                mesh.get_triangle(intersection.primId, &triangle);
                const auto &p = intersection.p;
                auto Ns = triangle.interpolated_normal(intersection.uv);
                Frame3f frame(Ns);
                auto w = cosine_hemisphere_sampling(sampler->next2d());
                w = frame.local_to_world(w);
                ray = Ray(p, w);
                ray.t_max = occlude_distance;
                if (scene->occlude(ray)) {
                    return Spectrum(0);
                } else {
                    return Spectrum(1);
                }
            } else {
                return Spectrum(0);
            }
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
                    //  if (spp <= 16) {
                    //     if (cur % (tot / 10) == 0) {
                    //         show_progress(double(cur) / tot, 70);
                    //     }
                    // } else if (spp <= 40) {
                    //     if (cur % (tot / 100) == 0) {
                    //         show_progress(double(cur) / tot, 70);
                    //     }

                    // } else {
                    //     if (cur % (tot / 200) == 0) {
                    //         show_progress(double(cur) / tot, 70);
                    //     }
                    // }
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

    class RTAO : public Integrator {
        [[refl]] int spp = 16;
        [[refl]] Float occlude_distance = 100000.0f;

      public:
        AKR_IMPLS(Integrator)
        bool supports_mode(RenderMode mode) const { return true; }
        std::shared_ptr<RenderTask> create_render_task(const RenderContext &ctx) override {
            return std::make_shared<AORenderTask>(ctx, spp, occlude_distance);
        }
    };
#include "generated/RTAO.hpp"
    AKR_EXPORT_PLUGIN(p) {}
} // namespace akari