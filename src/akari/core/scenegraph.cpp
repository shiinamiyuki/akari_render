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
namespace akari {
    namespace py = pybind11;
    std::vector<const char *> _get_enabled_variants() {
        return std::vector<const char *>(std::begin(enabled_variants), std::end(enabled_variants));
    }
    AKR_VARIANT class PerspectiveCameraNode : public CameraNode<Float, Spectrum> {
      public:
        AKR_IMPORT_BASIC_RENDER_TYPES()
        Point3f position;
        Vector3f rotation;
        Point2i resolution;
        Float fov = radians(80.0f);
        Camera compile() override {
            AKR_IMPORT_RENDER_TYPES(PerspectiveCamera)
            Transform3f c2w;
            c2w = Transform3f::rotate_z(rotation.z());
            c2w = Transform3f::rotate_x(rotation.y()) * c2w;
            c2w = Transform3f::rotate_y(rotation.x()) * c2w;
            c2w = Transform3f::translate(position) * c2w;
            return PerspectiveCamera(resolution, c2w, 1.0);
        }
    };
    AKR_VARIANT class OBJMesh : public MeshNode<Float, Spectrum> {
        std::string loaded;

      public:
        AKR_IMPORT_BASIC_RENDER_TYPES()
        AKR_IMPORT_RENDER_TYPES(SceneNode)
        OBJMesh() = default;
        OBJMesh(const std::string &path) : path(path) {}
        using MaterialNode = akari::MaterialNode<Float, Spectrum>;
        std::string path;
        Mesh mesh;
        std::vector<std::shared_ptr<MaterialNode>> materials;
        void commit() {
            if (loaded == path)
                return;
            (void)load_wavefront_obj(path);
        }
        MeshView compile(SceneNode &scene_node) {
            commit();
            MeshView view;
            view.indices = mesh.indices;
            view.material_indices = mesh.material_indices;
            view.normals = mesh.normals;
            view.texcoords = mesh.texcoords;
            view.vertices = mesh.vertices;
            scene_node.meshviews.push_back(view);
            return view;
        }

      private:
        bool load_wavefront_obj(const fs::path &obj) {
            info("loading {}\n", fs::absolute(path).string());
            loaded = obj.string();
            fs::path parent_path = fs::absolute(obj).parent_path();
            fs::path file = obj.filename();
            CurrentPathGuard _;
            if (!parent_path.empty())
                fs::current_path(parent_path);

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> obj_materials;

            std::string err;

            std::string _file = file.string();
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &obj_materials, &err, _file.c_str());
            (void)ret;
            for (size_t s = 0; s < shapes.size(); s++) {
                // Loop over faces(polygon)
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

                    // auto mat = materials[shapes[s].mesh.material_ids[f]].name;
                    int fv = shapes[s].mesh.num_face_vertices[f];
                    Point3f triangle[3];
                    for (int v = 0; v < fv; v++) {
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        mesh.indices.push_back(idx.vertex_index);
                        for (int i = 0; i < 3; i++) {
                            mesh.vertices.push_back(attrib.vertices[3 * idx.vertex_index + i]);
                        }
                        triangle[v] = load<Point3f>(&mesh.vertices[mesh.vertices.size() - 3]);
                    }

                    Normal3f ng =
                        normalize(cross(Vector3f(triangle[1] - triangle[0]), Vector3f(triangle[2] - triangle[1])));
                    for (int v = 0; v < fv; v++) {
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        if (idx.normal_index < 0) {
                            mesh.normals.emplace_back(ng[0]);
                            mesh.normals.emplace_back(ng[1]);
                            mesh.normals.emplace_back(ng[2]);
                        } else {
                            mesh.normals.emplace_back(attrib.normals[3 * idx.normal_index + 0]);
                            mesh.normals.emplace_back(attrib.normals[3 * idx.normal_index + 1]);
                            mesh.normals.emplace_back(attrib.normals[3 * idx.normal_index + 2]);
                        }
                        if (idx.texcoord_index < 0) {
                            mesh.texcoords.emplace_back(v > 0);
                            mesh.texcoords.emplace_back(v % 2 == 0);
                        } else {
                            mesh.texcoords.emplace_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
                            mesh.texcoords.emplace_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
                        }
                    }
                }
            }
            return true;
        }
    };

    AKR_VARIANT void register_scene_graph(py::module &parent) {
        auto m = parent.def_submodule(get_variant_string<Float, Spectrum>());
        using SceneGraphNode = akari::SceneGraphNode<Float, Spectrum>;
        using CameraNode = akari::CameraNode<Float, Spectrum>;
        using SceneNode = akari::SceneNode<Float, Spectrum>;
        using OBJMesh = akari::OBJMesh<Float, Spectrum>;
        using MeshNode = akari::MeshNode<Float, Spectrum>;
        using MaterialNode = akari::MaterialNode<Float, Spectrum>;
        using SceneNode = akari::SceneNode<Float, Spectrum>;
        using PerspectiveCameraNode = akari::PerspectiveCameraNode<Float, Spectrum>;
        py::class_<SceneGraphNode, std::shared_ptr<SceneGraphNode>>(m, "SceneGraphNode")
            .def("commit", &SceneGraphNode::commit);
        py::class_<SceneNode, SceneGraphNode, std::shared_ptr<SceneNode>>(m, "Scene")
            .def(py::init<>())
            .def_readwrite("variant", &SceneNode::variant)
            .def_readwrite("camera", &SceneNode::camera)
            .def_readwrite("shapes", &SceneNode::shapes)
            .def("commit", &SceneNode::commit);
        py::class_<CameraNode, SceneGraphNode, std::shared_ptr<CameraNode>>(m, "Camera");
        py::class_<PerspectiveCameraNode, CameraNode, std::shared_ptr<PerspectiveCameraNode>>(m, "PerspectiveCamera")
            .def(py::init<>())
            .def_readwrite("position", &PerspectiveCameraNode::position)
            .def_readwrite("rotation", &PerspectiveCameraNode::rotation)
            .def_readwrite("fov", &PerspectiveCameraNode::fov)
            .def("commit", &PerspectiveCameraNode::commit);
        py::class_<MaterialNode, SceneGraphNode, std::shared_ptr<MaterialNode>>(m, "Material");
        py::class_<MeshNode, SceneGraphNode, std::shared_ptr<MeshNode>>(m, "Mesh").def("commit", &MeshNode::commit);
        py::class_<OBJMesh, MeshNode, std::shared_ptr<OBJMesh>>(m, "OBJMesh")
            .def(py::init<>())
            .def(py::init<const std::string &>())
            .def("commit", &OBJMesh::commit)
            .def_readwrite("path", &OBJMesh::path);
        register_math_functions<Float, Spectrum>(m);
    }
    PYBIND11_EMBEDDED_MODULE(akari, m) {
        m.def("enabled_variants", _get_enabled_variants);
        for (auto &variant : enabled_variants) {
            AKR_INVOKE_VARIANT(std::string_view(variant), register_scene_graph, m);
        }
    }
} // namespace akari