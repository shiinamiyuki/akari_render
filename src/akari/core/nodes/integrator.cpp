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
#include <akari/core/nodes/integrator.h>
#include <akari/kernel/material.h>

namespace akari {
    AKR_VARIANT class AOIntegratorNode : public IntegratorNode<C> {
      public:
        AKR_IMPORT_TYPES()
        int spp = 16;
        int tile_size = 16;
        float occlude = std::numeric_limits<float>::infinity();
        std::shared_ptr<cpu::Integrator<C>> compile(MemoryArena<>*arena) override {
            return std::make_shared<cpu::Integrator<C>>(cpu::AmbientOcclusion<C>(spp, occlude));
        }
        const char *description() override { return "[Ambient Occlution]"; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "spp") {
                spp = value.get<int>().value();
            } else if (field == "occlude") {
                occlude = value.get<float>().value();
            }
        }
#ifdef AKR_ENABLE_GPU
        virtual std::shared_ptr<gpu::Integrator<C>> compile_gpu(MemoryArena<>*arena) {
            return std::make_shared<gpu::Integrator<C>>(gpu::AmbientOcclusion<C>(spp, occlude));
        }
#endif
    };
    AKR_VARIANT class PathIntegratorNode : public IntegratorNode<C> {
      public:
        AKR_IMPORT_TYPES()
        int spp = 16;
        int max_depth = 5;
        int tile_size = 256;
        float ray_clamp = 10.0f;
        std::shared_ptr<cpu::Integrator<C>> compile(MemoryArena<>*arena) override {
            return std::make_shared<cpu::Integrator<C>>(cpu::PathTracer<C>(spp));
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "spp") {
                spp = value.get<int>().value();
            } else if (field == "max_depth") {
                max_depth = value.get<int>().value();
            } else if (field == "tile_size") {
                tile_size = value.get<int>().value();
            } else if (field == "ray_clamp") {
                ray_clamp = value.get<float>().value();
            }
        }
#ifdef AKR_ENABLE_GPU
        std::shared_ptr<gpu::Integrator<C>> compile_gpu(MemoryArena<>*arena) override {
            return std::make_shared<gpu::Integrator<C>>(gpu::PathTracer<C>(spp, max_depth, tile_size, ray_clamp));
        }
#endif
        const char *description() override { return "[Path Tracer]"; }
    };

    AKR_VARIANT void RegisterIntegratorNode<C>::register_nodes() {
        AKR_IMPORT_TYPES()
        register_node<C, AOIntegratorNode<C>>("AO");
        register_node<C, PathIntegratorNode<C>>("Path");
    }

    AKR_VARIANT void RegisterIntegratorNode<C>::register_python_nodes(py::module &m) {
#ifdef AKR_ENABLE_PYTHON
        py::class_<IntegratorNode<C>, SceneGraphNode<C>, std::shared_ptr<IntegratorNode<C>>>(m, "Integrator");
        py::class_<AOIntegratorNode<C>, IntegratorNode<C>, std::shared_ptr<AOIntegratorNode<C>>>(m, "RTAO")
            .def(py::init<>())
            .def_readwrite("spp", &AOIntegratorNode<C>::spp)
            .def_readwrite("tile_size", &AOIntegratorNode<C>::tile_size)
            .def("commit", &AOIntegratorNode<C>::commit);
        py::class_<PathIntegratorNode<C>, IntegratorNode<C>, std::shared_ptr<PathIntegratorNode<C>>>(m, "Path")
            .def(py::init<>())
            .def_readwrite("spp", &PathIntegratorNode<C>::spp)
            .def_readwrite("tile_size", &PathIntegratorNode<C>::tile_size)
            .def("commit", &PathIntegratorNode<C>::commit);
#endif
    }

    AKR_RENDER_STRUCT(RegisterIntegratorNode)
} // namespace akari