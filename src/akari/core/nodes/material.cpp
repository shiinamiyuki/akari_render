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
#include <akari/core/nodes/material.h>
#include <akari/kernel/material.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace akari {
    AKR_VARIANT class DiffuseMaterialNode : public MaterialNode<C> {
      public:
        AKR_IMPORT_TYPES()
        Color3f color;
        Material<C> *compile(MemoryArena *arena) override {
            auto tex = arena->alloc<Texture<C>>(ConstantTexture<C>(color));
            return arena->alloc<Material<C>>(DiffuseMaterial<C>(tex));
        }
    };
    AKR_VARIANT void RegisterMaterialNode<C>::register_nodes(py::module &m) {
        AKR_IMPORT_TYPES()
        py::class_<MaterialNode<C>, SceneGraphNode<C>, std::shared_ptr<MaterialNode<C>>>(m, "Material");
        py::class_<DiffuseMaterialNode<C>, MaterialNode<C>, std::shared_ptr<DiffuseMaterialNode<C>>>(m,
                                                                                                     "DiffuseMaterial")
            .def(py::init<>())
            .def_readwrite("color", &DiffuseMaterialNode<C>::color)
            .def("commit", &DiffuseMaterialNode<C>::commit);
    }
    AKR_RENDER_STRUCT(RegisterMaterialNode)
} // namespace akari