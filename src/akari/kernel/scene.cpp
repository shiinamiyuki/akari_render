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

#include <akari/kernel/scene.h>
#include <akari/kernel/bvh-accelerator.h>
namespace akari {
    AKR_VARIANT bool Scene<C>::intersect(const Ray<C> &ray, Intersection<C> *intersection) const {
        bool hit = false;
        accel.dispatch([&](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, EmbreeAccelerator<C> *>) {
#ifndef AKR_GPU_CODE
                if constexpr (akari_enable_embree) {
                    hit = arg->intersect(ray, intersection);
                } else {
                    astd::abort();
                }
#else
                astd::abort();
#endif
            } else {
                hit = arg->intersect(ray, intersection);
            }
        });
        return hit;
    }
    AKR_VARIANT bool Scene<C>::occlude(const Ray<C> &ray) const {
        return accel.dispatch([&](auto &&arg) -> bool {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, EmbreeAccelerator<C> *>) {
#ifndef AKR_GPU_CODE
                if constexpr (akari_enable_embree) {
                    return arg->occlude(ray);
                } else {
                    astd::abort();
                }
#else
                astd::abort();
#endif
            } else {
                return arg->occlude(ray);
            }
        });
    }
    AKR_VARIANT void Scene<C>::commit() {
        accel.dispatch_cpu([&](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, EmbreeAccelerator<C> *>) {
#ifndef AKR_GPU_CODE
                if constexpr (akari_enable_embree) {
                    return arg->build(*this);
                } else {
                    astd::abort();
                }
#else
                astd::abort();
#endif
            } else {
                return arg->build(*this);
            }
        });
    }

    AKR_RENDER_CLASS(Scene)
} // namespace akari