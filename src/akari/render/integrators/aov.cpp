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
#include <akari/core/profiler.h>
#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/shaders/common.h>
namespace akari::render {
    class AOVIntegrator : public Integrator {
        const int tile_size = 16;

      public:
        enum AOV {
            albedo,
            normal,
        };
        int spp = 16;
        AOV aov;
        AOVIntegrator(int spp, AOV aov) : spp(spp), aov(aov) {}
        void render(const Scene *scene, Film *film) override {
            using namespace shader;
            AKR_ASSERT_THROW(glm::all(glm::equal(film->resolution(), scene->camera->resolution())));
            auto n_tiles = ivec2(film->resolution() + ivec2(tile_size - 1)) / ivec2(tile_size);
            debug("resolution: {}, tile size: {}, tiles: {}", film->resolution(), tile_size, n_tiles);
            std::mutex mutex;
            std::vector<astd::pmr::monotonic_buffer_resource *> resources;
            for (size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
                resources.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::get_default_resource()));
            }
            parallel_for_2d(n_tiles, [=, &mutex, &resources](const ivec2 &tile_pos, int tid) {
                Allocator<> allocator(resources[tid]);
                Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + ivec2(1)) * (int)tile_size};
                auto tile = film->tile(tileBounds);
                auto &camera = scene->camera;
                auto sampler = scene->sampler->clone(&allocator);
                for (int y = tile.bounds.pmin.y; y < tile.bounds.pmax.y; y++) {
                    for (int x = tile.bounds.pmin.x; x < tile.bounds.pmax.x; x++) {
                        sampler->set_sample_index(x + y * film->resolution().x);
                        for (int s = 0; s < spp; s++) {
                            sampler->start_next_sample();
                            CameraSample sample =
                                camera->generate_ray(sampler->next2d(), sampler->next2d(), ivec2(x, y));
                            Spectrum value;
                            if (auto isct = scene->intersect(sample.ray)) {
                                switch (aov) {
                                case AOV::albedo:
                                case AOV::normal:
                                    value = sample.normal;
                                    break;
                                }
                            }

                            tile.add_sample(vec2(x, y), value, 1.0f);
                        }
                    }
                }
                std::lock_guard<std::mutex> _(mutex);
                film->merge_tile(tile);
                resources[tid]->release();
            });
            for (auto rsrc : resources) {
                delete rsrc;
            }
        }
    };
    class AOVIntegratorNode final : public IntegratorNode {
      public:
        int spp = 16;
        AOVIntegrator::AOV aov = AOVIntegrator::albedo;
        Integrator *create_integrator(Allocator<> *alllocator) override {
            return alllocator->new_object<AOVIntegrator>(spp, aov);
        }
        const char *description() override { return "[Ambient Occlution]"; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "spp") {
                spp = value.get<int>().value();
            }
        }
        bool set_spp(int spp_) override {
            spp = spp_;
            return true;
        }
    };
    AKR_EXPORT_NODE(AOV, AOVIntegratorNode)

} // namespace akari::render