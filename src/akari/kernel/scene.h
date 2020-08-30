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
#include <akari/kernel/cameras/camera.h>
#include <akari/kernel/sampler.h>
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
        BufferView<MeshView> meshes;
        Camera<C> camera;
        Sampler<C> sampler;
        EmbreeAccelerator<C> *embree_scene = nullptr;
        bool intersect(const Ray3f &ray, Intersection<C> *isct) const;
        void commit();
    };
} // namespace akari
