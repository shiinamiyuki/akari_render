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
#include <akari/kernel/material.h>
#include <akari/kernel/sampling.h>
#include <akari/core/arena.h>
#include <akari/common/smallarena.h>
namespace akari {
    namespace cpu {

        AKR_VARIANT void AmbientOcclusion<C>::render(const Scene<C> &scene, Film<C> *film) const {
            AKR_ASSERT_THROW(all(film->resolution() == scene.camera.resolution()));
            auto n_tiles = Point2i(film->resolution() + Point2i(tile_size - 1)) / Point2i(tile_size);
            auto Li = [=, &scene](Ray3f ray, Sampler<C> &sampler) -> Spectrum {
                (void)scene;
                Intersection<C> intersection;
                if (scene.intersect(ray, &intersection)) {
                    Frame3f frame(intersection.ng);
                    // return Spectrum(intersection.ng);
                    auto w = sampling<C>::cosine_hemisphere_sampling(sampler.next2d());
                    w = frame.local_to_world(w);
                    ray = Ray3f(intersection.p, w);
                    intersection = Intersection<C>();
                    if (scene.intersect(ray, &intersection))
                        return Spectrum(0);
                    return Spectrum(1);
                }
                return Spectrum(0);
                // debug("{}\n", ray.d);
                // return Spectrum(ray.d * 0.5f + 0.5f);
            };
            debug("resolution: {}, tile size: {}, tiles: {}\n", film->resolution(), tile_size, n_tiles);
            std::mutex mutex;
            auto num_threads = num_work_threads();
            MemoryArena _arena;
            std::vector<SmallArena> small_arenas;
            for (auto i = 0u; i < num_threads; i++) {
                size_t size = 256 * 1024;
                small_arenas.emplace_back(_arena.alloc_bytes(size), size);
            }
            parallel_for_2d(n_tiles, [=, &scene, &mutex, &small_arenas](const Point2i &tile_pos, int tid) {
                (void)tid;
                Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + Vector2i(1)) * (int)tile_size};
                auto tile = film->tile(tileBounds);
                auto &camera = scene.camera;
                auto &arena = small_arenas[tid];
                (void)arena;
                auto sampler = scene.sampler;
                for (int y = tile.bounds.pmin.y(); y < tile.bounds.pmax.y(); y++) {
                    for (int x = tile.bounds.pmin.x(); x < tile.bounds.pmax.x(); x++) {
                        sampler.set_sample_index(x + y * film->resolution().x());
                        for (int s = 0; s < spp; s++) {
                            sampler.start_next_sample();
                            CameraSample<C> sample;
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

        AKR_VARIANT void PathTracer<C>::render(const Scene<C> &scene, Film<C> *film) const {
            AKR_ASSERT_THROW(all(film->resolution() == scene.camera.resolution()));
            auto n_tiles = Point2i(film->resolution() + Point2i(tile_size - 1)) / Point2i(tile_size);
            auto Li = [=, &scene](Ray3f ray, Sampler<C> &sampler, SmallArena *arena) -> Spectrum {
                (void)scene;
                int depth = 0;
                int max_depth = 5;
                Spectrum beta(1.0f);
                Spectrum L(0.0f);
                while (true) {
                    Intersection<C> intersection;
                    if (scene.intersect(ray, &intersection)) {
                        auto trig = scene.get_triangle(intersection.geom_id, intersection.prim_id);
                        SurfaceInteraction<C> si(intersection, trig);
                        MaterialEvalContext<C> ctx(sampler, si, arena);
                        auto wo = -ray.d;
                        auto *material = trig.material;
                        if (!material)
                            break;
                        if (material->template isa<EmissiveMaterial<C>>()) {
                            if (depth == 0) {
                                auto *emission = material->template get<EmissiveMaterial<C>>();
                                bool face_front = dot(ray.d, si.ng) < 0.0f;
                                if (emission->double_sided || face_front) {
                                    L += beta * emission->color->evaluate(ctx.texcoords);
                                }
                            }
                            break;
                        }
                        if (depth >= max_depth) {
                            break;
                        }
                        depth++;

                        si.bsdf = material->get_bsdf(ctx);
                        {
                            auto [light, light_pdf] = scene.select_light(sampler.next2d());
                            if (light) {
                                LightSampleContext<C> light_ctx;
                                light_ctx.u = sampler.next2d();
                                light_ctx.p = si.p;
                                LightSample<C> light_sample = light->sample(light_ctx);
                                light_pdf *= light_sample.pdf;
                                auto f = light_sample.L * si.bsdf.evaluate(wo, light_sample.wi) *
                                         std::abs(dot(si.ns, light_sample.wi));
                                if (!f.is_black()) {
                                    if (!scene.occlude(light_sample.test.ray)) {
                                        L += beta * f / light_pdf;
                                    }
                                }
                            }
                        }
                        BSDFSampleContext<C> sample_ctx(sampler.next2d(), -ray.d);
                        auto sample = si.bsdf.sample(sample_ctx);
                        beta *= sample.f * std::abs(dot(si.ng, sample.wi)) / sample.pdf;
                        ray = Ray3f(intersection.p, sample.wi, Constants<Float>::Eps() / std::abs(dot(si.ng, sample.wi)));

                    } else {
                        L += beta * Spectrum(0);
                        break;
                    }
                }
                return L;
            };
            std::mutex mutex;
            auto num_threads = num_work_threads();
            MemoryArena _arena;
            std::vector<SmallArena> small_arenas;
            for (auto i = 0u; i < num_threads; i++) {
                size_t size = 256 * 1024;
                small_arenas.emplace_back(_arena.alloc_bytes(size), size);
            }
            parallel_for_2d(n_tiles, [=, &scene, &mutex, &small_arenas](const Point2i &tile_pos, int tid) {
                (void)tid;
                Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + Vector2i(1)) * (int)tile_size};
                auto tile = film->tile(tileBounds);
                auto &camera = scene.camera;
                auto &arena = small_arenas[tid];
                auto sampler = scene.sampler;
                for (int y = tile.bounds.pmin.y(); y < tile.bounds.pmax.y(); y++) {
                    for (int x = tile.bounds.pmin.x(); x < tile.bounds.pmax.x(); x++) {
                        sampler.set_sample_index(x + y * film->resolution().x());
                        for (int s = 0; s < spp; s++) {
                            sampler.start_next_sample();
                            CameraSample<C> sample;
                            camera.generate_ray(sampler.next2d(), sampler.next2d(), Point2i(x, y), &sample);
                            auto L = Li(sample.ray, sampler, &arena);
                            tile.add_sample(Point2f(x, y), L, 1.0f);
                            arena.reset();
                        }
                    }
                }
                std::lock_guard<std::mutex> _(mutex);
                film->merge_tile(tile);
            });
        }
        AKR_RENDER_CLASS(AmbientOcclusion)
        AKR_RENDER_CLASS(PathTracer)
    } // namespace cpu
} // namespace akari
