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
#include <akari/core/python/scenegraph.h>
#include <akari/kernel/integrators/cpu/integrator.h>
#ifdef AKR_ENABLE_GPU
#    include <akari/kernel/integrators/gpu/integrator.h>
#endif
namespace akari {
    namespace gpu {
        AKR_VARIANT class Integrator;
    }
    AKR_VARIANT class IntegratorNode : public SceneGraphNode<C> {
      public:
        AKR_IMPORT_TYPES()
        virtual std::shared_ptr<cpu::Integrator<C>> compile(MemoryArena *arena) = 0;
        virtual std::shared_ptr<gpu::Integrator<C>> compile_gpu(MemoryArena *arena) { return nullptr; }
    };

    AKR_VARIANT struct RegisterIntegratorNode { static void register_nodes(py::module &m); };
} // namespace akari