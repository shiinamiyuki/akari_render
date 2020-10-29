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
#include <akari/core/math.h>
#include <akari/core/astd.h>
#include <akari/render/scenegraph.h>
#include <optional>
namespace akari::render {
    struct Intersection {
        Float t = Inf;
        Vec2 uv;
        int geom_id = -1;
        int prim_id = -1;
        bool is_instance = false;
        bool hit() const { return geom_id != -1 && prim_id != -1; }
    };
    class Scene;
    class Accelerator {
      public:
        virtual bool occlude(const Ray &ray) const = 0;
        virtual astd::optional<Intersection> intersect(const Ray &ray) const = 0;
        virtual void reset() = 0;
        virtual void build(const Scene &) = 0;
        virtual Bounds3f world_bounds() const = 0;
    };
    class AcceleratorNode : public SceneGraphNode {
      public:
        virtual std::shared_ptr<Accelerator> create_accel(const Scene &scene) = 0;
    };
} // namespace akari::render
