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

    class PathTracerIntegrator : public Integrator {
        int spp;
        int max_depth;
        const int tile_size = 16;

      public:
        PathTracerIntegrator(int spp, int max_depth) : spp(spp), max_depth(max_depth) {}
        void render(const Scene *scene, Film *film) override {
            info("Path Tracer");
            AKR_ASSERT_THROW(glm::all(glm::equal(film->resolution(), scene->camera->resolution())));
            auto n_tiles = ivec2(film->resolution() + ivec2(tile_size - 1)) / ivec2(tile_size);
            debug("resolution: {}, tile size: {}, tiles: {}", film->resolution(), tile_size, n_tiles);
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
            auto reporter = std::make_shared<ProgressReporter>(n_tiles.x * n_tiles.y, [=](size_t cur, size_t total) {
                bool show = (0 == cur % (estimate_tiles_per_sec));
                if (show) {
                    double tiles_per_sec = cur / std::max(1e-7, timer.elapsed_seconds());
                    double remaining = (total - cur) / tiles_per_sec;
                    show_progress(double(cur) / double(total), 60, timer.elapsed_seconds(), remaining);
                }
                if (cur == total) {
                    putchar('\n');
                }
            });
            parallel_for_2d(n_tiles, [=, &mutex, &resources](const ivec2 &tile_pos, int tid) {
                Allocator<> allocator(resources[tid]);
                Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + ivec2(1)) * (int)tile_size};
                auto tile = film->tile(tileBounds);
                auto camera = scene->camera;
                auto sampler = scene->sampler;
                for (int y = tile.bounds.pmin.y; y < tile.bounds.pmax.y; y++) {
                    for (int x = tile.bounds.pmin.x; x < tile.bounds.pmax.x; x++) {
                        sampler.set_sample_index(x + y * film->resolution().x);
                        for (int s = 0; s < spp; s++) {
                            sampler.start_next_sample();
                            GenericPathTracer pt;
                            pt.scene = scene;
                            pt.allocator = &allocator;
                            pt.sampler = &sampler;
                            pt.L = Spectrum(0.0);
                            pt.beta = Spectrum(1.0);
                            pt.max_depth = max_depth;
                            pt.run_megakernel(camera, ivec2(x, y));
                            tile.add_sample(vec2(x, y), min(clamp_zero(pt.L), Spectrum(10)), 1.0f);
                            resources[tid]->release();
                        }
                    }
                }
                std::lock_guard<std::mutex> _(mutex);
                film->merge_tile(tile);
                reporter->update();
            });
            for (auto rsrc : resources) {
                delete rsrc;
            }
        }
    };
    class PathIntegratorNode final : public IntegratorNode {
      public:
        int spp = 16;
        int max_depth = 5;
        Integrator *create_integrator(Allocator<> *allocator) override {
            return allocator->new_object<PathTracerIntegrator>(spp, max_depth);
        }
        const char *description() override { return "[Path Tracer]"; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "spp") {
                spp = value.get<int>().value();
            } else if (field == "max_depth") {
                max_depth = value.get<int>().value();
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