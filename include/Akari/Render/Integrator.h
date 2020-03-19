// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#include <Akari/Core/Component.h>
#include <Akari/Render/Accelerator.h>
#include <Akari/Render/Camera.h>
#include <Akari/Render/Light.h>
#include <Akari/Render/Material.h>
#include <Akari/Render/Mesh.h>
#include <Akari/Render/Sampler.h>
#include <Akari/Render/Scene.h>
#include <Akari/Render/Task.h>

namespace Akari {
    class RenderTask : public Task {
      public:
        enum class Event {
            ERENDER_DONE,
            EFILM_AVAILABLE
        };
        virtual bool HasFilmUpdate() = 0;
        virtual std::shared_ptr<const Film> GetFilmUpdate() = 0;
        virtual bool IsDone() = 0;
        virtual bool WaitEvent(Event event) = 0;
    };

    struct RenderContext {
        std::shared_ptr<const Scene> scene;
        std::shared_ptr<const Camera> camera;
        std::shared_ptr<const Sampler> sampler;
    };

    class Integrator : public Component {
      public:
        virtual std::shared_ptr<RenderTask> CreateRenderTask(const RenderContext &ctx) = 0;
    };

} // namespace Akari

#endif // AKARIRENDER_INTEGRATOR_H
