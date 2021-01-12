// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <akari/util.h>
#include <akari/pathtracer.h>
#include <spdlog/spdlog.h>
#include <akari/kdtree.h>
#include <akari/profile.h>
namespace akari::render {

    std::pair<Spectrum, Spectrum> render_pt_pixel_separete_emitter_direct(PTConfig config, Allocator<> allocator,
                                                                          const Scene &scene, Sampler &sampler,
                                                                          const vec2 &p_film) {
        pt::GenericPathTracer<pt::SeparateEmitPathVisitor> pt(&scene, &sampler, allocator, config.min_depth,
                                                              config.max_depth);
        pt.run_megakernel(&scene.camera.value(), p_film);
        AKR_ASSERT(hmax(pt.L) >= 0.0 && hmax(pt.emitter_direct) >= 0.0);
        return std::make_pair(pt.visitor.emitter_direct, pt.L);
    }
    Film render_pt(PTConfig config, const Scene &scene) {
        Film film(scene.camera->resolution());
        std::vector<astd::pmr::monotonic_buffer_resource *> buffers;
        for (size_t i = 0; i < thread::num_work_threads(); i++) {
            buffers.emplace_back(new astd::pmr::monotonic_buffer_resource(astd::pmr::new_delete_resource()));
        }
        ProgressReporter reporter(hprod(film.resolution()));
        thread::parallel_for(thread::blocked_range<2>(film.resolution(), ivec2(16, 16)), [&](ivec2 id, uint32_t tid) {
            Sampler sampler = config.sampler;
            sampler.set_sample_index(id.y * film.resolution().x + id.x);
            for (int s = 0; s < config.spp; s++) {
                sampler.start_next_sample();
                auto L = render_pt_pixel(config, Allocator<>(buffers[tid]), scene, sampler, id);
                buffers[tid]->release();
                film.add_sample(id, L, 1.0);
            }
            reporter.update();
        });
        for (auto buf : buffers) {
            delete buf;
        }
        spdlog::info("render pt done");
        return film;
    }
    namespace psd {
        struct IrradianceRecord {
            vec3 pos;
            vec3 wi;
            Spectrum irradiance;
            const vec3 &p() const { return pos; }
        };
        struct Tile {
            std::vector<IrradianceRecord> pts;
            bool computed = false;
            bool denoised = false;
            std::mutex m;
        };
    } // namespace psd
#if 0
    Image render_pt_psd(PTConfig config, PSDConfig psd_config, const Scene &scene) {
        using psd::IrradianceRecord;
        using psd::Tile;
        Film film(scene.camera->resolution());

        size_t tile_size = psd_config.filter_radius;

        ivec2 n_tiles = (film.resolution() + ivec2(tile_size) - ivec2(1)) / ivec2(tile_size);
        std::unique_ptr<Tile[]> tiles(new Tile[hprod(n_tiles)]);
        thread::ThreadPool pool(std::thread::hardware_concurrency());
        auto render_tile = [&](Tile &tile, ivec2 p) {
            std::lock_guard<std::mutex> _(tile.m);
            ivec2 upper = glm::min(p + ivec2(tile_size), film.resolution());
            astd::pmr::monotonic_buffer_resource rsrc(astd::pmr::new_delete_resource());
            for (int x = p.x; x < upper.x; x++) {
                for (int y = p.y; y < upper.y; y++) {
                    auto id = ivec2(x, y);
                    Sampler sampler = config.sampler;
                    sampler.set_sample_index(id.y * film.resolution().x + id.x);
                    for (int s = 0; s < config.spp; s++) {
                        sampler.start_next_sample();
                        // auto L = render_pt_pixel(config, Allocator<>(&rsrc), scene, sampler, id);
                        pt::GenericPathTracer ptracer;
                        ptracer.min_depth = config.min_depth;
                        ptracer.max_depth = config.max_depth;
                        ptracer.L = Spectrum(0.0);
                        ptracer.beta = Spectrum(1.0);
                        ptracer.sampler = &sampler;
                        ptracer.scene = &scene;
                        ptracer.allocator = Allocator<>(&rsrc);
                        {
                            auto camera_sample = ptracer.camera_ray(&scene.camera.value(), id);
                            Ray ray = camera_sample.ray;
                        }
                        rsrc.release();
                        film.add_sample(id, ptracer.L, 1.0);
                    }
                }
            }
        };
        spdlog::info("render pt done (with path space denoising)");
        return film.to_rgb_image();
    }
#endif
} // namespace akari::render