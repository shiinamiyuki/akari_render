// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <fstream>
#include <akari/scenegraph.h>
#include <spdlog/spdlog.h>
namespace akari::scene {
    Material::Material(){
        color.reset(new Texture());
        specular.reset(new Texture());
        metallic.reset(new Texture());
        roughness.reset(new Texture());
        emission.reset(new Texture());
    }
    static size_t MAGIC = 0x78567856;
    void Mesh::save_to_file(const std::string &file) const {
        std::ofstream out(file, std::ios::binary);
        size_t m = MAGIC;
        out.write((const char *)&m, sizeof(m));
        size_t vert_count = vertices.size();
        size_t face_count = indices.size();
        int8_t has_tc = !texcoords.empty();
        out.write((const char *)&vert_count, sizeof(vert_count));
        out.write((const char *)&face_count, sizeof(face_count));
        out.write((const char *)&has_tc, sizeof(has_tc));
        out.write((const char *)vertices.data(), sizeof(vec3) * vertices.size());
        out.write((const char *)normals.data(), sizeof(vec3) * normals.size());
        out.write((const char *)indices.data(), sizeof(ivec3) * indices.size());
        if (has_tc)
            out.write((const char *)texcoords.data(), sizeof(vec2) * texcoords.size());
        out.write((const char *)&m, sizeof(m));
    }
    void Mesh::load() {
        if (loaded)
            return;
        spdlog::info("loading {}", path);
        std::ifstream in(path, std::ios::binary);
        size_t m = 0;
        in.read((char *)&m, sizeof(m));
        AKR_ASSERT(m == MAGIC);
        size_t vert_count;
        size_t face_count;
        int8_t has_tc = false;
        in.read((char *)&vert_count, sizeof(vert_count));
        in.read((char *)&face_count, sizeof(face_count));
        in.read((char *)&has_tc, sizeof(has_tc));
        vertices.resize(vert_count);
        normals.resize(vert_count);
        indices.resize(face_count);
        in.read((char *)vertices.data(), sizeof(vec3) * vertices.size());
        in.read((char *)normals.data(), sizeof(vec3) * normals.size());
        in.read((char *)indices.data(), sizeof(ivec3) * indices.size());
        if (has_tc) {
            texcoords.resize(vert_count);
            in.read((char *)texcoords.data(), sizeof(vec2) * texcoords.size());
        }
        in.read((char *)&m, sizeof(m));
        AKR_ASSERT(m == MAGIC);
        loaded = true;
    }
    void Mesh::unload() {
        vertices = std::vector<vec3>();
        normals = std::vector<vec3>();
        texcoords = std::vector<vec2>();
        indices = std::vector<ivec3>();
        loaded = false;
    }
    void SceneGraph::commit() {
        for (auto &mesh : meshes) {
            mesh->load();
        }
    }
} // namespace akari::scene