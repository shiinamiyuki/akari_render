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
#include <akari/core/logger.h>
#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/common.h>
// #include <akari/render/denoiser.h>
namespace akari::render {
    RenderOutput UniAOVIntegrator::render(const RenderInput &input) {
        auto scene = input.scene;
        RenderOutput out;
        const auto resolution = scene->camera->resolution();
        out.aovs["color"].value = Film(resolution);
        do_render(scene, &out.aovs["color"].value.value());
        return out;
    }
    class AOVIntegrator : public UniAOVIntegrator {
        const int tile_size = 16;

      public:
        enum AOV {
            albedo,
            normal,
        };
        int spp = 16;
        AOV aov;
        AOVIntegrator(int spp, AOV aov) : spp(spp), aov(aov) {}
        void do_render(const Scene *scene, Film *film) override {
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
                auto sampler = scene->sampler->clone(Allocator<>());

                for (int y = tile.bounds.pmin.y; y < tile.bounds.pmax.y; y++) {
                    for (int x = tile.bounds.pmin.x; x < tile.bounds.pmax.x; x++) {
                        sampler->set_sample_index(x + y * film->resolution().x);
                        for (int s = 0; s < spp; s++) {
                            sampler->start_next_sample();
                            CameraSample sample =
                                camera->generate_ray(sampler->next2d(), sampler->next2d(), ivec2(x, y));
                            Spectrum value = Spectrum(0.0);
                            auto ray = sample.ray;
                            while (true) {
                                if (auto isct = scene->intersect(ray)) {
                                    auto trig = scene->get_triangle(isct->geom_id, isct->prim_id);
                                    Float u = sampler->next1d();
                                    Float tr = trig.material->tr(ShadingPoint(trig.texcoord(isct->uv)));
                                    if (tr > 0) {
                                        if (u < tr) {
                                            ray = Ray(trig.p(isct->uv), ray.d);
                                            continue;
                                        }
                                    }
                                    switch (aov) {
                                    case AOV::albedo: {
                                        auto mat = trig.material;
                                        ShadingPoint sp(trig.texcoord(isct->uv));
                                        value = mat->albedo(sp);
                                    } break;
                                    case AOV::normal:
                                        value = trig.ns(isct->uv);
                                        break;
                                    }
                                    break;
                                } else {
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
        std::shared_ptr<Integrator> create_integrator(Allocator<> allocator) override {
            return make_pmr_shared<AOVIntegrator>(allocator, spp, aov);
        }
        const char *description() override { return "[Ambient Occlution]"; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "spp") {
                spp = value.get<int>().value();
            }
            if (field == "aov") {
                auto v = value.get<std::string>().value();
                if (v == "albedo") {
                    aov = AOVIntegrator::albedo;
                } else if (v == "normal") {
                    aov = AOVIntegrator::normal;
                }
            }
        }
        bool set_spp(int spp_) override {
            spp = spp_;
            return true;
        }
        int get_spp() const override { return spp; }
    };
    AKR_EXPORT std::shared_ptr<IntegratorNode> make_aov_integrator() { return std::make_shared<AOVIntegratorNode>(); }
    AKR_EXPORT std::shared_ptr<IntegratorNode> make_aov_integrator(int spp, const char *aov_) {
        auto aov = std::string(aov_);
        auto integrator = std::make_shared<AOVIntegratorNode>();
        if (aov == "normal") {
            integrator->aov = AOVIntegrator::normal;
        } else if (aov == "albedo") {
            integrator->aov = AOVIntegrator::albedo;
        } else {
            error("invalid aov: {}", aov);
        }

        integrator->spp = spp;
        return integrator;
    }
} // namespace akari::render