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
#include <akari/core/scenegraph.h>
#include <akari/common/config.h>
namespace akari {
    namespace py = pybind11;
    std::vector<const char *> _get_enabled_variants() {
        return std::vector<const char *>(std::begin(enabled_variants), std::end(enabled_variants));
    }

    class OBJMesh : public MeshNode {
      public:
        std::string path;
    };
    void register_scene_graph(py::module &m) {
        py::class_<SceneGraphNode, std::shared_ptr<SceneGraphNode>>(m, "SceneGraphNode")
            .def("commit", &SceneGraphNode::commit);
        py::class_<SceneNode, SceneGraphNode, std::shared_ptr<SceneNode>>(m, "Scene")
            .def(py::init<>())
            .def_readwrite("variant", &SceneNode::variant);
        py::class_<CameraNode, SceneGraphNode, std::shared_ptr<CameraNode>>(m, "Camera").def(py::init<>());
        py::class_<MaterialNode, SceneGraphNode, std::shared_ptr<MaterialNode>>(m, "Material").def(py::init<>());
        py::class_<MeshNode, SceneGraphNode, std::shared_ptr<MeshNode>>(m, "Mesh")
            .def(py::init<>())
            .def_readwrite("material", &MeshNode::material)
            .def("commit", &MeshNode::commit);
        py::class_<OBJMesh, MeshNode, std::shared_ptr<OBJMesh>>(m, "OBJMesh")
            .def(py::init<>())
            .def("commit", &OBJMesh::commit)
            .def_readwrite("path", &OBJMesh::path);
        m.def("enabled_variants", _get_enabled_variants);
    }
    PYBIND11_EMBEDDED_MODULE(akari, m) { register_scene_graph(m); }
} // namespace akari