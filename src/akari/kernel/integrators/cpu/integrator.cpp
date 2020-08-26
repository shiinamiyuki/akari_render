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
#include <akari/kernel/integrators/cpu/integrator.h>
#include <akari/core/film.h>
#include <akari/core/logger.h>
#include <akari/kernel/scene.h>
#include <akari/kernel/interaction.h>
namespace akari {
    namespace cpu {
        AKR_VARIANT void AmbientOcclusion<Float, Spectrum>::render(const AScene &scene, AFilm *film) const {
            AKR_ASSERT_THROW(all(film->resolution() == scene.camera.resolution()));
            AKR_IMPORT_RENDER_TYPES(Intersection, SurfaceInteraction)
            auto n_tiles = Point2i(film->resolution() + Point2i(tile_size - 1)) / Point2i(tile_size);
            auto Li = [=, &scene](const Ray3f &ray, ASampler &sampler) -> Spectrum {
                (void)scene;
                AIntersection intersection;
                if (scene.intersect(ray, &intersection)) {
                    // debug("{}\n", intersection.t / 20.0f);
                    return Spectrum(intersection.ng);
                }
                return Spectrum(0);
                // debug("{}\n", ray.d);
                // return Spectrum(ray.d * 0.5f + 0.5f);
            };
            debug("resolution: {}, tile size: {}, tiles: {}\n", film->resolution(), tile_size, n_tiles);
            std::mutex mutex;
            parallel_for_2d(n_tiles, [=, &scene, &mutex](const Point2i &tile_pos, int tid) {
                (void)tid;
                Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + Vector2i(1)) * (int)tile_size};
                auto tile = film->tile(tileBounds);
                auto sampler = scene.sampler;
                auto &camera = scene.camera;
                for (int y = tile.bounds.pmin.y(); y < tile.bounds.pmax.y(); y++) {
                    for (int x = tile.bounds.pmin.x(); x < tile.bounds.pmax.x(); x++) {
                        sampler.set_sample_index(x + y * film->resolution().x());
                        for (int s = 0; s < 1; s++) {
                            sampler.start_next_sample();
                            ACameraSample sample;
                            camera.generate_ray(sampler.next2d(), sampler.next2d(), Point2i(x, y), &sample);
                            auto L = Li(sample.ray, sampler);
                            tile.add_sample(Point2f(x, y), L, 1.0f);
                        }
                    }
                }
                std::lock_guard<std::mutex> _(mutex);
                film->merge_tile(tile);
            });
        }
        AKR_RENDER_CLASS(AmbientOcclusion)
    } // namespace cpu
} // namespace akari
