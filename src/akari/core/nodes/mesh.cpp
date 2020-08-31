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
#include <akari/core/nodes/mesh.h>
#include <akari/core/resource.h>
#include <akari/core/mesh.h>
// #define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <akari/core/logger.h>

namespace akari {
    AKR_VARIANT class AkariMesh : public MeshNode<C> {
      public:
        std::string path;
        AkariMesh() = default;
        AkariMesh(std::string path) : path(path) {}
        std::shared_ptr<Mesh> mesh;
        std::vector<std::shared_ptr<MaterialNode<C>>> materials;
        void commit() override {
            auto exp = resource_manager()->load_path<BinaryGeometry>(path);
            if (exp) {
                mesh = exp.extract_value()->mesh();
            } else {
                auto err = exp.extract_error();
                error("error loading {}: {}\n", path, err.what());
                std::abort();
            }
        }
        MeshView compile(MemoryArena *) override {
            commit();
            MeshView view;
            view.indices = mesh->indices;
            view.material_indices = mesh->material_indices;
            view.normals = mesh->normals;
            view.texcoords = mesh->texcoords;
            view.vertices = mesh->vertices;
            return view;
        }
    };
    AKR_VARIANT class OBJMesh : public MeshNode<C> {
        std::string loaded;

      public:
        AKR_IMPORT_TYPES()
        OBJMesh() = default;
        OBJMesh(const std::string &path) : path(path) {}
        using MaterialNode = akari::MaterialNode<C>;
        std::string path;
        Mesh mesh;
        std::vector<std::shared_ptr<MaterialNode>> materials;
        void commit() override {
            if (loaded == path)
                return;
            (void)load_wavefront_obj(path);
        }
        MeshView compile(MemoryArena *) override {
            commit();
            MeshView view;
            view.indices = mesh.indices;
            view.material_indices = mesh.material_indices;
            view.normals = mesh.normals;
            view.texcoords = mesh.texcoords;
            view.vertices = mesh.vertices;
            return view;
        }

      private:
        bool load_wavefront_obj(const fs::path &obj) {
            info("loading {}\n", fs::absolute(obj).string());
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
            mesh.vertices.resize(attrib.vertices.size());
            std::memcpy(&mesh.vertices[0], &attrib.vertices[0], sizeof(float) * mesh.vertices.size());
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
                        triangle[v] = Point3f(load<PackedArray<Float, 3>>(&mesh.vertices[3 * idx.vertex_index]));
                    }

                    Normal3f ng =
                        normalize(cross(Vector3f(triangle[1] - triangle[0]), Vector3f(triangle[2] - triangle[0])));
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

                    index_offset += fv;
                }
            }
            info("loaded {} triangles, {} vertices\n", mesh.indices.size() / 3, mesh.vertices.size() / 3);
            return true;
        }
    };
    AKR_VARIANT void RegisterMeshNode<C>::register_nodes(py::module &m) {
        AKR_IMPORT_TYPES();
        py::class_<MeshNode<C>, SceneGraphNode<C>, std::shared_ptr<MeshNode<C>>>(m, "Mesh").def("commit",
                                                                                                &MeshNode<C>::commit);
        py::class_<OBJMesh<C>, MeshNode<C>, std::shared_ptr<OBJMesh<C>>>(m, "OBJMesh")
            .def(py::init<>())
            .def(py::init<const std::string &>())
            .def("commit", &OBJMesh<C>::commit)
            .def_readwrite("path", &OBJMesh<C>::path);
        py::class_<AkariMesh<C>, MeshNode<C>, std::shared_ptr<AkariMesh<C>>>(m, "AkariMesh")
            .def(py::init<>())
            .def(py::init<const std::string &>())
            .def("commit", &AkariMesh<C>::commit)
            .def_readwrite("path", &AkariMesh<C>::path);
        m.def("load_mesh", [](const std::string &path) { return std::make_shared<AkariMesh<C>>(path); });
    }
    AKR_RENDER_STRUCT(RegisterMeshNode)
} // namespace akari