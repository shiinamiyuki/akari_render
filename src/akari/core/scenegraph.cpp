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

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <akari/core/python.h>
// #define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <akari/core/scenegraph.h>
#include <akari/common/config.h>
#include <akari/core/mesh.h>
#include <akari/kernel/scene.h>
#include <akari/core/logger.h>
#include <akari/core/film.h>
#include <akari/core/nodes/camera.h>
#include <akari/core/nodes/mesh.h>
#include <akari/core/nodes/scene.h>
namespace akari {
    namespace py = pybind11;

    AKR_VARIANT void RegisterSceneGraph<C>::register_scene_graph(py::module &parent) {
        auto m = parent.def_submodule(get_variant_string<C>());
        AKR_IMPORT_TYPES();

        RegisterMathFunction<C>::register_math_functions(m);
        py::class_<SceneGraphNode<C>, std::shared_ptr<SceneGraphNode<C>>>(m, "SceneGraphNode")
            .def("commit", &SceneGraphNode<C>::commit);
        py::class_<MaterialNode<C>, SceneGraphNode<C>, std::shared_ptr<MaterialNode<C>>>(m, "Material");
        RegisterSceneNode<C>::register_nodes(m);
        RegisterCameraNode<C>::register_nodes(m);
        RegisterMeshNode<C>::register_nodes(m);

        m.def("set_device_cpu", &set_device_cpu);
        m.def("set_device_gpu", &set_device_gpu);
    }
    AKR_RENDER_STRUCT(RegisterSceneGraph)
} // namespace akari