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
#include <akari/common/math.h>
#include <akari/kernel/meshview.h>
#include <akari/kernel/camera.h>
#include <akari/kernel/sampler.h>
#include <akari/kernel/shape.h>
#include <akari/kernel/light.h>
#ifdef AKR_ENABLE_EMBREE
#    include <akari/kernel/embree.inl>
#endif

namespace akari {
    AKR_VARIANT
    class EmbreeAccelerator;
    AKR_VARIANT struct Intersection {
        AKR_IMPORT_TYPES()
        Point3f p;
        Float t;
        Normal3f ng;
        Point2f uv;
        int geom_id = -1;
        int prim_id = -1;
        bool is_instance = false;
    };
    AKR_VARIANT class Scene {
      public:
        AKR_IMPORT_TYPES()
        BufferView<MeshView<C>> meshes;
        Camera<C> camera;
        Sampler<C> sampler;
        BufferView<AreaLight<C>> area_lights;
        EmbreeAccelerator<C> *embree_scene = nullptr;

        AKR_CPU bool intersect(const Ray3f &ray, Intersection<C> *isct) const;
        AKR_CPU bool occlude(const Ray3f &ray) const;

        void commit();
        AKR_XPU Triangle<C> get_triangle(int mesh_id, int prim_id) const {
            auto &mesh = meshes[mesh_id];
            Triangle<C> trig = get_triangle(mesh, prim_id);
            auto mat_idx = mesh.material_indices[prim_id];
            if (mat_idx != -1) {
                trig.material = mesh.materials[mat_idx];
            }
            return trig;
        }
        AKR_XPU astd::pair<const AreaLight<C> *, Float> select_light(const Point2f &u) const {
            if (area_lights.size() == 0) {
                return {nullptr, Float(0.0f)};
            }
            size_t idx = area_lights.size() * u[0];
            if (idx == area_lights.size()) {
                idx -= 1;
            }
            return {&area_lights[idx], Float(1.0 / area_lights.size())};
        }
    };
} // namespace akari
