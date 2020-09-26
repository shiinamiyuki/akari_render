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
#include <akari/kernel/sampler.h>
#include <akari/kernel/camera.h>
#include <akari/kernel/pathtracer.h>

namespace akari {
    AKR_VARIANT struct PathState {
        AKR_IMPORT_TYPES()
        Sampler<C> sampler;
        Spectrum L;
        Spectrum beta;
        AKR_XPU GenericPathTracer<C> path_tracer() const {
            GenericPathTracer<C> pt;
            pt.sampler = sampler;
            pt.L = L;
            pt.beta = beta;
            return pt;
        }
        AKR_XPU void update(const GenericPathTracer<C> &pt) {
            sampler = pt.sampler;
            L = pt.L;
            beta = pt.beta;
        }
        // bool terminated = false;
    };
    AKR_VARIANT struct RayWorkItem {
        AKR_IMPORT_TYPES()
        int pixel;
        Ray3f ray;
        bool valid() const { return ray.tmax > 0; }
    };

    template <typename C>
    struct MaterialWorkItem {
        AKR_IMPORT_TYPES()
        int pixel;
        const Material<C> *material = nullptr;
        int geom_id = -1;
        int prim_id = -1;
        Point2f uv;
        Vector3f wo;
        AKR_XPU SurfaceHit<C> surface_hit() const {
            SurfaceHit<C> hit;
            hit.geom_id = geom_id;
            hit.prim_id = prim_id;
            hit.uv = uv;
            hit.wo = wo;
            hit.material = material;
            return hit;
        }
    };

    AKR_VARIANT struct ShadowRayWorkItem {
        AKR_IMPORT_TYPES()
        int pixel;
        Ray3f ray;
        Spectrum color;
        Float pdf = 0.0f;
        AKR_XPU bool hit() const { return ray.tmax < 0.0f; }
        ShadowRayWorkItem() = default;
        ShadowRayWorkItem(const DirectLighting<C> &lighting) {
            color = lighting.color;
            ray = lighting.shadow_ray;
            pdf = lighting.pdf;
        }
        DirectLighting<C> direct_lighting() const {
            DirectLighting<C> lighting;
            lighting.color = color;
            lighting.shadow_ray = ray;
            lighting.pdf = pdf;
            return lighting;
        }
    };
} // namespace akari