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

#include <akari/core/serialize.hpp>
#include <akari/plugins/binary_mesh.h>
#include <akari/plugins/matte.h>
#include <akari/plugins/rgbtexture.h>
#include <akari/render/scene_graph.h>
#include <memory>
#include <iostream>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace akari {
    std::shared_ptr<Mesh> Convert(const fs::path &obj, fs::path output) {
        std::shared_ptr<Mesh> mesh;
        if (output.extension().empty()) {
            output.concat(".mesh");
        }
        {
            fs::path parent_path = fs::absolute(obj).parent_path();
            fs::path file = obj.filename();
            if (!parent_path.empty())
                fs::current_path(parent_path);

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;

            std::string err;

            std::string _file = file.string();
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, _file.c_str());
            (void)ret;
            std::vector<Vertex> vertices;
            std::vector<int> indices;
            std::unordered_map<std::string, int> name_to_group;
            std::vector<MaterialSlot> cvtMaterials;
            std::vector<int> group;
            for (size_t s = 0; s < shapes.size(); s++) {
                // Loop over faces(polygon)
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

                    auto mat = materials[shapes[s].mesh.material_ids[f]].name;

                    auto name = shapes[s].name + "_" + mat;

                    if (name_to_group.find(name) == name_to_group.end()) {
                        name_to_group[name] = name_to_group.size();

                        // Convert material
                        {
                            auto material = materials[shapes[s].mesh.material_ids[f]];
                            MaterialSlot cvtMaterial;
                            vec3 kd = vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
                            vec3 ke = vec3(material.emission[0], material.emission[1], material.emission[2]);
                            Float strength = 1;
                            if (MaxComp(ke) > 0) {
                                strength = MaxComp(ke);
                                ke /= strength;
                            }
                            if (MinComp(ke) > 0.01) {
                                cvtMaterial.marked_as_light = true;
                            } else {
                                cvtMaterial.marked_as_light = false;
                            }
                            cvtMaterial.material = create_matte_material(create_rgb_texture(kd));
                            cvtMaterial.name = name;
                            cvtMaterial.emission.strength = create_rgb_texture(vec3(strength));
                            cvtMaterial.emission.color = create_rgb_texture(ke);
                            cvtMaterials.emplace_back(std::move(cvtMaterial));
                        }
                    }

                    group.push_back(name_to_group[name]);

                    int fv = shapes[s].mesh.num_face_vertices[f];

                    Vertex vertex[3];
                    for (int v = 0; v < fv; v++) {
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                        indices.push_back(indices.size());
                        for (int i = 0; i < 3; i++) {
                            vertex[v].pos[i] = attrib.vertices[3 * idx.vertex_index + i];
                        }
                    }
                    auto Ng = normalize(cross(vertex[1].pos - vertex[0].pos, vertex[2].pos - vertex[0].pos));
                    for (int v = 0; v < fv; v++) {
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
                    // per-face material
                    // shapes[s].mesh.material_ids[f];
                }
            }
            mesh = create_binary_mesh(file,std::move(vertices),std::move(indices),std::move(group),std::move(cvtMaterials));
        }
        mesh->save_path(output.string().c_str());
        return mesh;
    }
} // namespace akari

int main(int argc, char **argv) {
    using namespace akari;
    if (argc < 2) {
        fprintf(stderr, "Usage: AkariConvert input-filename output-filename ");
        exit(-1);
    }
    fs::path in = argv[1];
    fs::path out = argc >= 3 ? argv[2] : fs::path(in).replace_extension("");

    std::shared_ptr<SceneGraph> graph;
    CurrentPathGuard _guard;
    auto mesh = Convert(in, out);
    if(out.extension().empty()){
        out.concat(".json");
    }
    auto data = serialize::save_to_json(mesh);
    std::ofstream o(out);
    o << data.dump(1) << std::endl;
    std::cout << "convert done\n";
    return 0;
}