// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#include <Akari/Render/Plugins/BinaryMesh.h>
#include <Akari/Render/SceneGraph.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace Akari {
    bool Convert(const fs::path &obj, const fs::path &binary, SceneGraph *graph = nullptr) {
        std::shared_ptr<BinaryMesh> mesh;
        {
            fs::path parent_path = fs::absolute(obj).parent_path();
            fs::path file = obj.filename();
            CurrentPathGuard _guard;
            if (!parent_path.empty())
                fs::current_path(parent_path);

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;

            std::string err;

            std::string _file = file.string();
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, _file.c_str());
            std::vector<Vertex> vertices;
            std::vector<ivec3> indices;

            for (size_t s = 0; s < shapes.size(); s++) {
                // Loop over faces(polygon)
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

                    auto mat = materials[shapes[s].mesh.material_ids[f]].name;

                    int fv = shapes[s].mesh.num_face_vertices[f];

                    Vertex vertex[3];
                    ivec3 index;
                    for (size_t v = 0; v < fv; v++) {
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        index[v] = int(indices.size() + v);
                        for (int i = 0; i < 3; i++) {
                            vertex[v].pos[i] = attrib.vertices[3 * idx.vertex_index + i];
                        }
                    }
                    auto Ng = normalize(cross(vertex[1].pos - vertex[0].pos, vertex[2].pos - vertex[0].pos));
                    for (auto &i : vertex) {
                        i.Ng = Ng;
                    }
                    for (size_t v = 0; v < fv; v++) {
                        // access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        if (idx.normal_index >= 0) {
                            for (int i = 0; i < 3; i++)
                                vertex[v].Ns[i] = attrib.normals[3 * idx.normal_index + i];
                        } else {
                            vertex[v].Ns = Ng;
                        }
                        if (idx.texcoord_index >= 0) {
                            for (int i = 0; i < 2; i++)
                                vertex[v].texCoord[i] = attrib.texcoords[2 * idx.texcoord_index + i];
                        } else {
                            vertex[v].texCoord = vec2(v > 0, v & 0);
                        }
                    }
                    index_offset += fv;
                    for (auto &i : vertex) {
                        vertices.emplace_back(i);
                    }
                    indices.emplace_back(index);
                    // per-face material
                    // shapes[s].mesh.material_ids[f];
                }
            }
            mesh = std::make_shared<BinaryMesh>();
            mesh->file = binary.string();
            mesh->vertexBuffer = std::move(vertices);
            mesh->indexBuffer = std::move(indices);
        }
        mesh->Save(binary.string().c_str());
        if (graph) {
            graph->meshes.emplace_back(mesh);
        }
        return true;
    }
} // namespace Akari

int main(int argc, char ** argv) {
    using namespace Akari;
    if(argc < 2) {
        fprintf(stderr, "Usage: MeshImporter input-filename output-filename [scene-filename]");
        exit(-1);
    }
    fs::path in = argv[1];
    fs::path out = argv[2];
    std::shared_ptr<SceneGraph> graph;

}