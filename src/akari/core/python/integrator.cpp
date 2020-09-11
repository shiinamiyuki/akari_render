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
#include <akari/core/python/integrator.h>
#include <akari/kernel/material.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace akari {
    AKR_VARIANT class AOIntegratorNode : public IntegratorNode<C> {
      public:
        AKR_IMPORT_TYPES()
        int spp = 16;
        cpu::Integrator<C> *compile(MemoryArena *arena) override {
            return arena->alloc<cpu::Integrator<C>>(cpu::AmbientOcclusion<C>(spp));
        }
    };
    AKR_VARIANT class PathIntegratorNode : public IntegratorNode<C> {
      public:
        AKR_IMPORT_TYPES()
        int spp = 16;
        cpu::Integrator<C> *compile(MemoryArena *arena) override {
            return arena->alloc<cpu::Integrator<C>>(cpu::PathTracer<C>(spp));
        }
    };
    AKR_VARIANT void RegisterIntegratorNode<C>::register_nodes(py::module &m) {
        AKR_IMPORT_TYPES()
        py::class_<IntegratorNode<C>, SceneGraphNode<C>, std::shared_ptr<IntegratorNode<C>>>(m, "Integrator");
        py::class_<AOIntegratorNode<C>, IntegratorNode<C>, std::shared_ptr<AOIntegratorNode<C>>>(m, "RTAO")
            .def(py::init<>())
            .def_readwrite("spp", &AOIntegratorNode<C>::spp)
            .def("commit", &AOIntegratorNode<C>::commit);
        py::class_<PathIntegratorNode<C>, IntegratorNode<C>, std::shared_ptr<PathIntegratorNode<C>>>(m, "Path")
            .def(py::init<>())
            .def_readwrite("spp", &PathIntegratorNode<C>::spp)
            .def("commit", &PathIntegratorNode<C>::commit);
    }
    AKR_RENDER_STRUCT(RegisterIntegratorNode)
} // namespace akari