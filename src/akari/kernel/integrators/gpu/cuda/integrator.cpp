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
            size_t ARENA_SIZE = 16 * 1024;

            MemoryArena _arena;
            SmallArena arena(_arena.alloc_bytes(ARENA_SIZE), ARENA_SIZE);
            debug("GPU RTAO resolution: {}, tile size: {}, tiles: {}", film->resolution(), tile_size, n_tiles);
            double gpu_time = 0;
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
                    Timer timer;
                    launch(
                        "RTAO", extents.x() * extents.y(), AKR_GPU_LAMBDA(int tid) {
                            int tx = tid % extents.x();
                            int ty = tid / extents.x();
                            int x = tx + tileBounds.pmin.x();
                            int y = ty + tileBounds.pmin.y();
                            sampler.set_sample_index(x + y * resolution.x());
                            Spectrum acc;
                            for (int s = 0; s < spp; s++) {
                                sampler.start_next_sample();
                                CameraSample<C> sample;

                                camera.generate_ray(sampler.next2d(), sampler.next2d(), Point2i(x, y), &sample);
                                auto L = Li(sample.ray, sampler);
                                acc += L;
                            }
                            tile->add_sample(Point2f(x, y), acc / spp, 1.0f);
                        });
                    CUDA_CHECK(cudaDeviceSynchronize());
                    gpu_time += timer.elapsed_seconds();
                    film->merge_tile(*tile);
                }
            }
            info("total gpu time {}s", gpu_time);
        } else {
            fatal("only float is supported for gpu");
        }
    }
    AKR_RENDER_CLASS(AmbientOcclusion)
    AKR_VARIANT struct PathTracerImpl {
        AKR_IMPORT_TYPES()
        using Allocator = astd::pmr::polymorphic_allocator<>;
        using RayQueue = WorkQueue<RayWorkItem<C>>;
        int spp;
        int tile_size = 512;
        int max_depth = 5;
        size_t MAX_QUEUE_SIZE;
        SOA<PathState<C>> path_states;
        RayQueue *ray_queue[2] = {nullptr, nullptr};
        using MaterialQueue = WorkQueue<MaterialWorkItem<C>>;
        using MaterialWorkQueues = astd::array<MaterialQueue *, Material<C>::num_types>;
        MaterialWorkQueues material_queues;
        PathTracerImpl(Allocator &allocator, const PathTracer<C> &pt) : spp(pt.spp), tile_size(pt.tile_size) {
            MAX_QUEUE_SIZE = tile_size * tile_size;
            path_states = SOA<PathState<C>>(MAX_QUEUE_SIZE, allocator);
            ray_queue[0] = allocator.template new_object<RayQueue>(MAX_QUEUE_SIZE, allocator);
            ray_queue[1] = allocator.template new_object<RayQueue>(MAX_QUEUE_SIZE, allocator);
            for (size_t i = 0; i < material_queues.size(); i++) {
                material_queues[i] = allocator.template new_object<MaterialQueue>(MAX_QUEUE_SIZE, allocator);
            }
        }
        void render(const Scene<C> &scene, Film<C> *film);
    };
    AKR_VARIANT void PathTracer<C>::render(const Scene<C> &scene, Film<C> *film) const {
        if constexpr (std::is_same_v<Float, float>) {
            auto resource = std::make_unique<auto_release_resource>(get_device_memory_resource());
            auto allocator = astd::pmr::polymorphic_allocator(resource.get());
            PathTracerImpl<C> tracer(allocator, *this);
            tracer.render(scene, film);
        } else {
            fatal("only float is supported for gpu");
        }
    }

    AKR_VARIANT void PathTracerImpl<C>::render(const Scene<C> &scene, Film<C> *film) {
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
            auto get_pixel = AKR_GPU_LAMBDA(int pixel_id) {
                int tx = pixel_id % extents.x();
                int ty = pixel_id / extents.x();
                int x = tx + tileBounds.pmin.x();
                int y = ty + tileBounds.pmin.y();
                return Point2i(x, y);
            };
            auto launch_size = extents.x() * extents.y();
            Timer timer;
            launch(
                "Set Path State", launch_size, AKR_GPU_LAMBDA(int tid) {
                    auto px = get_pixel(tid);
                    int x = px.x(), y = px.y();
                    auto sampler = _sampler;
                    sampler.set_sample_index(x + y * resolution.x());
                    PathState<C> path_state = path_states[tid];
                    path_state.sampler = sampler;
                    path_state.L = Spectrum(0);
                    path_states[tid] = path_state;
                });
            for (int s = 0; s < _spp; s++) {
                launch(
                    "Generate Camera Ray", launch_size, AKR_GPU_LAMBDA(int tid) {
                        auto px = get_pixel(tid);
                        int x = px.x(), y = px.y();
                        PathState<C> path_state = path_states[tid];
                        path_state.L = Spectrum(0);
                        path_state.beta = Spectrum(1.0f);
                        auto &sampler = path_state.sampler;
                        sampler.start_next_sample();
                        CameraSample<C> sample;
                        camera.generate_ray(sampler.next2d(), sampler.next2d(), Point2i(x, y), &sample);
                        RayWorkItem<C> ray_item = (*ray_queue[0])[tid];
                        ray_item.pixel = tid;
                        ray_item.ray = sample.ray;
                        (*ray_queue[0])[tid] = ray_item;
                        path_states[tid] = path_state;
                    });
                // first bounce ray queue is full
                launch_single(
                    "Set Ray Queue", AKR_GPU_LAMBDA() { ray_queue[0]->head = MAX_QUEUE_SIZE; });
                for (int depth = 0; depth <= max_depth; depth++) {
                    for (size_t i = 0; i < material_queues.size(); i++) {
                        launch_single(
                            "Reset Material Queue", AKR_GPU_LAMBDA() { material_queues[i]->clear(); });
                    }
                    launch(
                        "Intersect Closest", launch_size, AKR_GPU_LAMBDA(int tid) {
                            if (tid >= ray_queue[0]->elements_in_queue())
                                return;
                            RayWorkItem<C> ray_item = (*ray_queue[0])[tid];
                            if (!ray_item.valid()) {
                                return;
                            }

                            Intersection<C> intersection;
                            if (scene.intersect(ray_item.ray, &intersection)) {
                                // auto trig = scene.get_triangle(intersection.geom_id, intersection.prim_id);
                                auto mat_idx =
                                    scene.meshes[intersection.geom_id].material_indices[intersection.prim_id];
                                if (mat_idx < 0)
                                    return;

                                auto *material = scene.meshes[intersection.geom_id].materials[mat_idx];
                                if (!material)
                                    return;
                                MaterialWorkItem<C> material_item;
                                material_item.pixel = ray_item.pixel;
                                material_item.material = material;
                                material_item.geom_id = intersection.geom_id;
                                material_item.prim_id = intersection.prim_id;
                                material_item.uv = intersection.uv;
                                material_item.wo = -ray_item.ray.d;
                                auto queue_idx = material->typeindex();
                                AKR_ASSERT(queue_idx >= 0 && queue_idx < material_queues.size());
                                material_queues[queue_idx]->append(material_item);
                            }
                        });

                    launch_single(
                        "Reset Ray Queue", AKR_GPU_LAMBDA() { ray_queue[0]->clear(); });
                    for (auto material_queue : material_queues) {
                        launch(
                            "Evaluate Material", launch_size, AKR_GPU_LAMBDA(int tid) {
                                if (tid >= material_queue->elements_in_queue()) {
                                    return;
                                }
                                int cur_depth = depth;
                                MaterialWorkItem<C> material_item = (*material_queue)[tid];
                                int pixel = material_item.pixel;
                                PathState<C> path_state = path_states[pixel];
                                auto &sampler = path_state.sampler;

                                auto trig = scene.get_triangle(material_item.geom_id, material_item.prim_id);
                                SurfaceInteraction<C> si(material_item.uv, trig);
                                MaterialEvalContext<C> ctx(sampler, si);
                                auto *material = material_item.material;
                                auto wo = material_item.wo;
                                if (material->template isa<EmissiveMaterial<C>>()) {
                                    if (true || depth == 0) {
                                        auto *emission = material->template get<EmissiveMaterial<C>>();
                                        bool face_front = dot(-wo, si.ng) < 0.0f;
                                        if (emission->double_sided || face_front) {
                                            path_state.L += path_state.beta * emission->color->evaluate(ctx.texcoords);
                                        }
                                    }
                                } else if (cur_depth < max_depth) {
                                    si.bsdf = material->get_bsdf(ctx);
                                    cur_depth++;
                                    BSDFSampleContext<C> sample_ctx(sampler.next2d(), wo);
                                    auto sample = si.bsdf.sample(sample_ctx);
                                    path_state.beta *= sample.f * std::abs(dot(si.ng, sample.wi)) / sample.pdf;
                                    auto ray = Ray3f(si.p, sample.wi,
                                                     Constants<Float>::Eps() / std::abs(dot(si.ng, sample.wi)));
                                    RayWorkItem<C> ray_item;
                                    ray_item.pixel = pixel;
                                    ray_item.ray = ray;
                                    ray_queue[0]->append(ray_item);
                                }
                                path_states[pixel] = path_state;
                            });
                    }
                }
                launch(
                    "Update Film", launch_size, AKR_GPU_LAMBDA(int tid) {
                        PathState<C> state = path_states[tid];
                        auto p = get_pixel(tid);
                        tile->add_sample(Point2f(p), state.L, 1.0f);
                    });
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            info("tile took {}s", timer.elapsed_seconds());
            film->merge_tile(*tile);
        };
        auto n_tiles = Point2i(film->resolution() + Point2i(tile_size - 1)) / Point2i(tile_size);
        for (int tile_y = 0; tile_y < n_tiles.y(); tile_y++) {
            for (int tile_x = 0; tile_x < n_tiles.x(); tile_x++) {
                render_tile(tile_x, tile_y);
            }
        }
        print_kernel_stats();
    }
    AKR_RENDER_CLASS(PathTracer)

} // namespace akari::gpu