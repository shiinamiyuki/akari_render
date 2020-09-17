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

#include <akari/core/parallel.h>
#include <akari/kernel/integrators/gpu/integrator.h>
#include <akari/core/film.h>
#include <akari/core/logger.h>
#include <akari/kernel/scene.h>
#include <akari/kernel/interaction.h>
#include <akari/kernel/material.h>
#include <akari/kernel/sampling.h>
#include <akari/core/arena.h>
#include <akari/common/smallarena.h>
#include <akari/kernel/cuda/launch.h>
namespace akari::gpu {
    AKR_VARIANT void AmbientOcclusion<C>::render(const Scene<C> &scene, Film<C> *film) const {
        if constexpr (std::is_same_v<Float, float>) {
            AKR_ASSERT_THROW(all(film->resolution() == scene.camera.resolution()));
            auto n_tiles = Point2i(film->resolution() + Point2i(tile_size - 1)) / Point2i(tile_size);
            auto Li = AKR_GPU_LAMBDA(Ray3f ray, Sampler<C> & sampler)->Spectrum {
                Intersection<C> intersection;
                if (scene.intersect(ray, &intersection)) {
                    Frame3f frame(intersection.ng);
                    auto w = sampling<C>::cosine_hemisphere_sampling(sampler.next2d());
                    w = frame.local_to_world(w);
                    ray = Ray3f(intersection.p, w);
                    intersection = Intersection<C>();
                    if (scene.intersect(ray, &intersection))
                        return Spectrum(0);
                    return Spectrum(1);
                }
                return Spectrum(0);
            };
            size_t ARENA_SIZE = 16 * 1024;
            MemoryArena _arena;
            SmallArena arena(_arena.alloc_bytes(ARENA_SIZE), ARENA_SIZE);
            debug("GPU RTAO resolution: {}, tile size: {}, tiles: {}\n", film->resolution(), tile_size, n_tiles);
            for (int tile_y = 0; tile_y < n_tiles.y(); tile_y++) {
                for (int tile_x = 0; tile_x < n_tiles.x(); tile_x++) {
                    Point2i tile_pos(tile_x, tile_y);
                    Bounds2i tileBounds =
                        Bounds2i{tile_pos * (int)tile_size, (tile_pos + Vector2i(1)) * (int)tile_size};
                    auto boxed_tile = film->boxed_tile(tileBounds);
                    auto &camera = scene.camera;
                    auto sampler = scene.sampler;
                    auto extents = tileBounds.extents();
                    auto tile = boxed_tile.get();
                    auto resolution = film->resolution();
                    launch(
                        "RTAO", extents.x() * extents.y(), AKR_GPU_LAMBDA(int tid) {
                            int tx = tid % extents.x();
                            int ty = tid / extents.x();
                            int x = tx + tileBounds.pmin.x();
                            int y = ty + tileBounds.pmin.y();
                            sampler.set_sample_index(x + y * resolution.x());
                            for (int s = 0; s < spp; s++) {
                                sampler.start_next_sample();
                                CameraSample<C> sample;

                                camera.generate_ray(sampler.next2d(), sampler.next2d(), Point2i(x, y), &sample);
                                auto L = Li(sample.ray, sampler);
                                tile->add_sample(Point2f(x, y), L, 1.0f);
                            }
                        });
                    CUDA_CHECK(cudaDeviceSynchronize());
                    film->merge_tile(*tile);
                }
            }
        } else {
            fatal("only float is supported for gpu\n");
        }
    }
    AKR_RENDER_CLASS(AmbientOcclusion)

    AKR_VARIANT struct CameraRayWorkItem {
        AKR_IMPORT_CORE_TYPES_WITH(float)
        CameraSample<C> sample;
    };

    AKR_VARIANT void PathTracer<C>::render(const Scene<C> &scene, Film<C> *film) const {
        if constexpr (std::is_same_v<Float, float>) {
            // for (int tile_y = 0; tile_y < n_tiles.y(); tile_y++) {
            //     for (int tile_x = 0; tile_x < n_tiles.x(); tile_x++) {
            //         Point2i tile_pos(tile_x, tile_y);
            //         Bounds2i tileBounds =
            //             Bounds2i{tile_pos * (int)tile_size, (tile_pos + Vector2i(1)) * (int)tile_size};
            //         auto boxed_tile = film->boxed_tile(tileBounds);
            //         auto tile = boxed_tile.get();
            //     }
            // }
        } else {
            fatal("only float is supported for gpu\n");
        }
    }
    AKR_RENDER_CLASS(PathTracer)

} // namespace akari::gpu