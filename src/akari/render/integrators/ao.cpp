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
#include <akari/render/common.h>
namespace akari::render {
    class AmbientOcclusion : public UniAOVIntegrator {
        const int tile_size = 16;
        int spp = 16;
        float occlude = std::numeric_limits<float>::infinity();

      public:
        AmbientOcclusion(int spp, float occlude) : spp(spp), occlude(occlude) {}
        void do_render(const Scene *scene, Film *film) override {

            AKR_ASSERT_THROW(glm::all(glm::equal(film->resolution(), scene->camera->resolution())));
            auto n_tiles = ivec2(film->resolution() + ivec2(tile_size - 1)) / ivec2(tile_size);
            auto Li = [=](Ray ray, Sampler &sampler) -> Spectrum {
                (void)scene;
                if (auto intersection = scene->intersect(ray)) {
                    auto trig = scene->get_triangle(intersection->geom_id, intersection->prim_id);
                    Frame frame(trig.ng());
                    auto w = cosine_hemisphere_sampling(sampler.next2d());
                    w = frame.local_to_world(w);
                    ray = Ray(trig.p(intersection->uv), w);
                    if (auto r = scene->intersect(ray)) {
                        if (r->t < occlude) {
                            return Spectrum(0);
                        }
                    }
                    // debug("{} {}",trig.ng(),trig.ng() * 0.5f + 0.5f );
                    return Spectrum(1);
                }
                return Spectrum(0);
            };
            debug("resolution: {}, tile size: {}, tiles: {}", film->resolution(), tile_size, n_tiles);
            std::mutex mutex;
            std::vector<astd::pmr::monotonic_buffer_resource *> resources;
            for (size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
                resources.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::get_default_resource()));
            }
            thread::parallel_for(
                thread::blocked_range<2>(n_tiles), [=, &mutex, &resources](const ivec2 &tile_pos, int tid) {
                    Allocator<> allocator(resources[tid]);
                    Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + ivec2(1)) * (int)tile_size};
                    auto tile = film->tile(tileBounds);
                    auto &camera = scene->camera;
                    auto sampler = scene->sampler->clone(Allocator<>());
                    for (int y = tile.bounds.pmin.y; y < tile.bounds.pmax.y; y++) {
                        for (int x = tile.bounds.pmin.x; x < tile.bounds.pmax.x; x++) {
                            sampler->set_sample_index(x + y * film->resolution().x);
                            for (int s = 0; s < spp; s++) {
                                sampler->start_next_sample();
                                CameraSample sample =
                                    camera->generate_ray(sampler->next2d(), sampler->next2d(), ivec2(x, y));
                                auto L = Li(sample.ray, *sampler.get());
                                tile.add_sample(vec2(x, y), L, 1.0f);
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
    class AOIntegratorNode final : public IntegratorNode {
      public:
        AKR_SER_CLASS("RTAO")
        int spp = 16;
        int tile_size = 16;
        float occlude = std::numeric_limits<float>::infinity();
        std::shared_ptr<Integrator> create_integrator(Allocator<> allocator) override {
            return make_pmr_shared<AmbientOcclusion>(allocator, spp, occlude);
        }
        const char *description() override { return "[Ambient Occlution]"; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "spp") {
                spp = value.get<int>().value();
            } else if (field == "occlude") {
                occlude = value.get<float>().value();
            }
        }
        bool set_spp(int spp_) override {
            spp = spp_;
            return true;
        }
        int get_spp() const override { return spp; }
    };
    AKR_EXPORT_NODE(RTAO, AOIntegratorNode)

} // namespace akari::render