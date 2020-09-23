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
#include <sstream>
#include <pugixml.hpp>
#include <akari/core/akari.h>
#include <akari/common/fwd.h>
#include <akari/common/buffer.h>
#include <akari/core/arena.h>
#include <akari/kernel/instance.h>
#include <akari/kernel/integrators/cpu/integrator.h>

#ifdef AKR_ENABLE_PYTHON
#    include <pybind11/pybind11.h>
#    include <pybind11/embed.h>
#    include <pybind11/stl.h>
#endif
// #include <akari/kernel/material.h>
namespace pybind11 {
    class module;
}
namespace akari {
    namespace py = pybind11;
    class Node {
      public:
        virtual void load(const pugi::xml_node &xml) = 0;
        virtual void commit() {}
        virtual const char *description() { return "unknown"; }
        virtual ~Node() = default;
        typedef std::shared_ptr<Node> (*CreateFunc)(void);
    };
    AKR_VARIANT class SceneGraphNode : public Node {
      public:
        using Float = typename C::Float;
        AKR_IMPORT_CORE_TYPES()
    };
    AKR_VARIANT class SceneNode;

    AKR_VARIANT class FilmNode : public SceneGraphNode<C> { public: };
    AKR_VARIANT struct RegisterSceneGraph { static void register_scene_graph(py::module &parent); };

    AKR_EXPORT std::shared_ptr<Node> create_node_with_name(std::string_view variant, const std::string &name);
    AKR_EXPORT void register_node(std::string_view variant, const std::string &name, Node::CreateFunc);
    AKR_VARIANT void register_node(const std::string &name, Node::CreateFunc f) {
        register_node(get_variant_string<C>(), name, f);
    }
    template <class C, class T>
    void register_node(const std::string &name) {
        register_node(get_variant_string<C>(), name, []() -> std::shared_ptr<Node> { return std::make_shared<T>(); });
    }
} // namespace akari