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

#pragma once
#include <akari/common/fwd.h>
#include <akari/common/variant.h>
#include <akari/kernel/scene.h>
namespace akari {
    namespace gpu {
        AKR_VARIANT class AmbientOcclusion {
          public:
            int spp = 16;
            const int tile_size = 1024;
            float occlude = std::numeric_limits<float>::infinity();
            AKR_IMPORT_TYPES()
            AmbientOcclusion() = default;
            AmbientOcclusion(int spp, float occlude) : spp(spp), occlude(occlude) {}
            void render(const Scene<C> &scene, Film<C> *out) const;
        };
        AKR_VARIANT class PathTracer {
          public:
            template <typename C1>
            friend struct PathTracerImpl;
            int spp = 16;
            int max_depth = 5;
            int tile_size = 256;
            float ray_clamp = 0;
            bool wavefront = true;
            AKR_IMPORT_TYPES()
            PathTracer() = default;
            PathTracer(int spp, int max_depth, int tile_size, float ray_clamp, bool wavefront)
                : spp(spp), max_depth(max_depth), tile_size(tile_size), ray_clamp(ray_clamp), wavefront(wavefront) {}
            void render(const Scene<C> &scene, Film<C> *out) const;
        };
        AKR_VARIANT class Integrator : public Variant<AmbientOcclusion<C>, PathTracer<C>> {
          public:
            AKR_IMPORT_TYPES()
            using Variant<AmbientOcclusion<C>, PathTracer<C>>::Variant;
            void render(const Scene<C> &scene, Film<C> *out) const { AKR_VAR_DISPATCH(render, scene, out); }
        };
    } // namespace gpu
} // namespace akari