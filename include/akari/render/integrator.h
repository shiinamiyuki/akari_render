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

#ifndef AKARIRENDER_INTEGRATOR_H
#define AKARIRENDER_INTEGRATOR_H

#include <akari/core/component.h>
#include <akari/render/accelerator.h>
#include <akari/render/camera.h>
#include <akari/render/light.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/sampler.h>
#include <akari/render/scene.h>
#include <akari/render/task.h>

namespace akari {
    class RenderTask : public Task {
      public:
        enum class Event { ERENDER_DONE, EFILM_AVAILABLE };
        virtual bool has_film_update() = 0;
        virtual std::shared_ptr<const Film> film_update() = 0;
        virtual bool is_done() = 0;
        virtual bool wait_event(Event event) = 0;
    };

    struct RenderContext {
        std::shared_ptr<const Scene> scene;
        std::shared_ptr<const Camera> camera;
        std::shared_ptr<const Sampler> sampler;
    };

    class Integrator : public Component {
      public:
        enum class RenderMode {
          EProgressive,
          ETile,
        };
        virtual bool supports_mode(RenderMode mode) const = 0;
        virtual std::shared_ptr<RenderTask> create_render_task(const RenderContext &ctx) = 0;

    };
} // namespace akari

#endif // AKARIRENDER_INTEGRATOR_H
