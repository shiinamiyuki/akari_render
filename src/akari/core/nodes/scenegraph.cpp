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
#ifdef AKR_ENABLE_PYTHON
#    include <pybind11/pybind11.h>
#    include <pybind11/embed.h>
#    include <pybind11/stl.h>
#    include <akari/core/nodes/python.h>
#endif
// #define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <akari/core/nodes/scenegraph.h>
#include <akari/common/config.h>
#include <akari/core/mesh.h>
#include <akari/kernel/scene.h>
#include <akari/core/logger.h>
#include <akari/core/film.h>
#include <akari/core/nodes/camera.h>
#include <akari/core/nodes/mesh.h>
#include <akari/core/nodes/material.h>
#include <akari/core/nodes/integrator.h>
#include <akari/core/nodes/scene.h>
namespace akari {
    namespace __node_register {
        static std::unordered_map<std::string_view, std::unordered_map<std::string, Node::CreateFunc>> map;
    }
    std::shared_ptr<Node> create_node_with_name(std::string_view variant, const std::string &name) {
        using namespace __node_register;
        auto vit = map.find(variant);
        if (map.end() == vit) {
            throw std::runtime_error(fmt::format("variant {} not found", variant.data()));
        }
        auto it = vit->second.find(name);
        if (it == vit->second.end()) {
            throw std::runtime_error(fmt::format("{}.{} is not registered", variant.data(), name));
        }
        return it->second();
    }
    void register_node(std::string_view variant, const std::string &name, Node::CreateFunc func) {
        using namespace __node_register;
        map[variant][name] = func;
    }

    namespace py = pybind11;

    AKR_VARIANT void RegisterSceneGraph<C>::register_python_scene_graph(py::module &parent) {
#ifdef AKR_ENABLE_PYTHON
        auto m = parent.def_submodule(get_variant_string<C>());
        AKR_IMPORT_TYPES();

        RegisterMathFunction<C>::register_math_functions(m);
        py::class_<SceneGraphNode<C>, std::shared_ptr<SceneGraphNode<C>>>(m, "SceneGraphNode")
            .def("commit", &SceneGraphNode<C>::commit);
        RegisterSceneNode<C>::register_python_nodes(m);
        RegisterCameraNode<C>::register_python_nodes(m);
        RegisterMeshNode<C>::register_python_nodes(m);
        RegisterMaterialNode<C>::register_python_nodes(m);
        RegisterIntegratorNode<C>::register_python_nodes(m);
        m.def("set_device_cpu", []() {
            warning("compute device set to cpu\n");
            warning("use akari [scene file] instead\n");
            set_device_cpu();
        });
        m.def("set_device_gpu", []() {
            warning("compute device set to gpu\n");
            warning("use akari --gpu [scene file] instead\n");
            set_device_gpu();
        });
        m.def("get_device", []() -> std::string { return get_device() == ComputeDevice::cpu ? "cpu" : "gpu"; });
#endif
    }

    AKR_VARIANT void RegisterSceneGraph<C>::register_scene_graph() {
        RegisterSceneNode<C>::register_nodes();
        RegisterCameraNode<C>::register_nodes();
        RegisterMeshNode<C>::register_nodes();
        RegisterMaterialNode<C>::register_nodes();
        RegisterIntegratorNode<C>::register_nodes();
    }

    AKR_RENDER_STRUCT(RegisterSceneGraph)

} // namespace akari
