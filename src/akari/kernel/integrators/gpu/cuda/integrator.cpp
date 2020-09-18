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
#include <akari/kernel/cuda/workqueue.h>
#include <akari/kernel/integrators/gpu/workitem-soa.h>
#include <akari/core/profiler.h>
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
            auto resource = std::make_unique<auto_release_resource>(get_device_memory_resource());
            auto allocator = astd::pmr::polymorphic_allocator(resource.get());
            size_t MAX_QUEUE_SIZE = tile_size * tile_size;
            auto *camera_ray_queue =
                allocator.template new_object<WorkQueue<CameraRayWorkItem<C>>>(MAX_QUEUE_SIZE, allocator);
            auto *path_states = allocator.template new_object<WorkQueue<PathState<C>>>(MAX_QUEUE_SIZE, allocator);
            auto *ray_queue = allocator.template new_object<WorkQueue<RayWorkItem<C>>>(MAX_QUEUE_SIZE, allocator);
            auto *shadow_ray_queue =
                allocator.template new_object<WorkQueue<ShadowRayWorkItem<C>>>(MAX_QUEUE_SIZE, allocator);
            auto *closest_hit_queue =
                allocator.template new_object<WorkQueue<ClosestHitWorkItem<C>>>(MAX_QUEUE_SIZE, allocator);
            auto *any_hit_queue =
                allocator.template new_object<WorkQueue<AnyHitWorkItem<C>>>(MAX_QUEUE_SIZE, allocator);
            auto *miss_queue = allocator.template new_object<WorkQueue<MissWorkItem<C>>>(MAX_QUEUE_SIZE, allocator);
            auto render_tile = [&](int tile_x, int tile_y) {
                Point2i tile_pos(tile_x, tile_y);
                Bounds2i tileBounds = Bounds2i{tile_pos * (int)tile_size, (tile_pos + Vector2i(1)) * (int)tile_size};
                auto boxed_tile = film->boxed_tile(tileBounds);
                auto &camera = scene.camera;
                auto _sampler = scene.sampler;
                auto extents = tileBounds.extents();
                auto tile = boxed_tile.get();
                auto resolution = film->resolution();
                auto _spp = this->spp;
                Timer timer;
                auto get_pixel = AKR_GPU_LAMBDA(int pixel_id) {
                    int tx = pixel_id % extents.x();
                    int ty = pixel_id / extents.x();
                    int x = tx + tileBounds.pmin.x();
                    int y = ty + tileBounds.pmin.y();
                    return Point2i(x, y);
                };
                size_t launch_size = extents.x() * extents.y();
                launch(
                    "set_path_state", launch_size, AKR_GPU_LAMBDA(int tid) {
                        int tx = tid % extents.x();
                        int ty = tid / extents.x();
                        int x = tx + tileBounds.pmin.x();
                        int y = ty + tileBounds.pmin.y();
                        auto sampler = _sampler;
                        sampler.set_sample_index(x + y * resolution.x());
                        PathState<C> path_state = (*path_states)[tid];
                        path_state.sampler = sampler;
                        path_state.pixel = tid;
                        path_state.L = Spectrum(0);
                        (*path_states)[tid] = path_state;
                    });
                for (int s = 0; s < _spp; s++) {
                    launch(
                        "generate_camera_ray", launch_size, AKR_GPU_LAMBDA(int tid) {
                            int tx = tid % extents.x();
                            int ty = tid / extents.x();
                            int x = tx + tileBounds.pmin.x();
                            int y = ty + tileBounds.pmin.y();
                            PathState<C> path_state = (*path_states)[tid];
                            auto &sampler = path_state.sampler;
                            sampler.start_next_sample();
                            CameraSample<C> sample;
                            camera.generate_ray(sampler.next2d(), sampler.next2d(), Point2i(x, y), &sample);
                            CameraRayWorkItem<C> work_item;
                            work_item.sample = sample;
                            (*camera_ray_queue)[tid] = work_item;
                             path_state.L = Spectrum(0);
                            (*path_states)[tid] = path_state;
                            RayWorkItem<C> ray_work_item;
                            ray_work_item.ray = sample.ray;
                            ray_work_item.pixel = tid;
                            (*ray_queue)[tid] = ray_work_item;
                        });
                    launch_single(
                        "reset_closest_hit", AKR_GPU_LAMBDA(int) {
                            closest_hit_queue->clear();
                            miss_queue->clear();
                            any_hit_queue->clear();
                            shadow_ray_queue->clear();
                        });
                    launch(
                        "closest_hit", launch_size, AKR_GPU_LAMBDA(int tid) {
                            RayWorkItem<C> ray_work_item = (*ray_queue)[tid];
                            ClosestHitWorkItem<C> closest_hit;
                            closest_hit.pixel = ray_work_item.pixel;
                            if (scene.intersect(ray_work_item.ray, &closest_hit.intersection)) {
                                (*closest_hit_queue).append(closest_hit);
                            } else {
                                MissWorkItem<C> miss;
                                miss.pixel = ray_work_item.pixel;
                                miss_queue->append(miss);
                            }
                        });
#if 0
                    launch(
                        "on_miss", launch_size, AKR_GPU_LAMBDA(int tid) {
                            if (tid >= miss_queue->elements_in_queue())
                                return;
                            MissWorkItem<C> miss = (*miss_queue)[tid];
                            // auto pixel = get_pixel(closest_hit.pixel);
                            PathState<C> state = (*path_states)[miss.pixel];
                            state.L = Spectrum(0);
                            (*path_states)[miss.pixel] = state;
                        });
#endif
                    launch(
                        "on_closest_hit", launch_size, AKR_GPU_LAMBDA(int tid) {
                            if (tid >= closest_hit_queue->elements_in_queue())
                                return;
                            ClosestHitWorkItem<C> closest_hit = (*closest_hit_queue)[tid];
                            PathState<C> state = (*path_states)[closest_hit.pixel];
                            auto &intersection = closest_hit.intersection;
                            Frame3f frame(intersection.ng);
                            auto w = sampling<C>::cosine_hemisphere_sampling(state.sampler.next2d());
                            w = frame.local_to_world(w);
                            ShadowRayWorkItem<C> shadow_ray;
                            shadow_ray.pixel = closest_hit.pixel;
                            shadow_ray.ray = Ray3f(intersection.p, w);
                            shadow_ray_queue->append(shadow_ray);
                            // state.L = intersection.ng * 0.5f + 0.5f;
                            (*path_states)[closest_hit.pixel] = state;
                        });

                    launch(
                        "any_hit", launch_size, AKR_GPU_LAMBDA(int tid) {
                            if (tid >= shadow_ray_queue->elements_in_queue())
                                return;
                            ShadowRayWorkItem<C> shadow_ray = (*shadow_ray_queue)[tid];
                            AnyHitWorkItem<C> any_hit;
                            any_hit.pixel = shadow_ray.pixel;
                            any_hit.hit = scene.occlude(shadow_ray.ray);
                            any_hit_queue->append(any_hit);
                        });
                    launch(
                        "on_any_hit_miss", launch_size, AKR_GPU_LAMBDA(int tid) {
                            if (tid >= any_hit_queue->elements_in_queue())
                                return;
                            AnyHitWorkItem<C> any_hit = (*any_hit_queue)[tid];
                            // auto pixel = get_pixel(closest_hit.pixel);
                            if (!any_hit.hit) {
                                PathState<C> state = (*path_states)[any_hit.pixel];
                                state.L = Spectrum(1);
                                (*path_states)[any_hit.pixel] = state;
                            }
                        });

                    launch(
                        "update_film", launch_size, AKR_GPU_LAMBDA(int tid) {
                            PathState<C> state = (*path_states)[tid];
                            auto p = get_pixel(tid);
                            tile->add_sample(Point2f(p), state.L, 1.0f);
                        });
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                info("tile took {}s\n", timer.elapsed_seconds());
                film->merge_tile(*tile);
            };
            for (int tile_y = 0; tile_y < n_tiles.y(); tile_y++) {
                for (int tile_x = 0; tile_x < n_tiles.x(); tile_x++) {
                    render_tile(tile_x, tile_y);
                }
            }
#if 0
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
#endif
        } else {
            fatal("only float is supported for gpu\n");
        }
    }
    AKR_RENDER_CLASS(AmbientOcclusion)

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