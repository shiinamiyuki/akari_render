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

#include <akari/core/logger.h>
#include <akari/core/plugin.h>
#include <akari/core/stream.h>
#include <akari/plugins/area_light.h>
#include <akari/plugins/binary_mesh.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <fstream>
namespace akari {
    const char *AKR_MESH_MAGIC = "AKARI_BINARY_MESH";
    Expected<bool> BinaryGeometry::save(const fs::path &path) const {
        std::ofstream out(path, std::ios::binary | std::ios::out);
        out.write(AKR_MESH_MAGIC, strlen(AKR_MESH_MAGIC));
        size_t vertexCount = vertex_buffer.size();
        size_t triangleCount = groups.size();
        out.write(reinterpret_cast<const char *>(&vertexCount), sizeof(size_t));
        out.write(reinterpret_cast<const char *>(&triangleCount), sizeof(size_t));
        out.write(reinterpret_cast<const char *>(vertex_buffer.data()), sizeof(Vertex) * vertexCount);
        out.write(reinterpret_cast<const char *>(index_buffer.data()), sizeof(int) * triangleCount * 3);
        out.write(reinterpret_cast<const char *>(groups.data()), sizeof(int) * triangleCount);
        out.write(AKR_MESH_MAGIC, strlen(AKR_MESH_MAGIC));
        return true;
    }
    Expected<bool> BinaryGeometry::load(const fs::path &path) {
        info("Loading {}\n", path.string());
        std::ifstream in(path, std::ios::binary | std::ios::in);
        char buffer[128] = {0};
        in.read(buffer, strlen(AKR_MESH_MAGIC));
        if (strcmp(buffer, AKR_MESH_MAGIC) != 0) {
            error("Failed to load mesh: invalid format\n");
            return Error("Invalid format");
        }
        size_t vertexCount;
        size_t triangleCount;
        in.read(reinterpret_cast<char *>(&vertexCount), sizeof(size_t));
        in.read(reinterpret_cast<char *>(&triangleCount), sizeof(size_t));
        vertex_buffer.resize(vertexCount);
        in.read(reinterpret_cast<char *>(vertex_buffer.data()), sizeof(Vertex) * vertexCount);
        index_buffer.resize(triangleCount * 3);
        in.read(reinterpret_cast<char *>(index_buffer.data()), sizeof(int) * vertexCount);
        groups.resize(triangleCount);
        in.read(reinterpret_cast<char *>(groups.data()), sizeof(int) * groups.size());
        memset(buffer, 0, sizeof(buffer));
        in.read(buffer, strlen(AKR_MESH_MAGIC));
        if (strcmp(buffer, AKR_MESH_MAGIC) != 0) {
            error("Failed to load mesh: invalid format\n");
            return Error("Invalid format");
        }
        info("Loaded {} triangles\n", groups.size());
        return true;
    }

    class AKR_EXPORT BinaryMesh : public Mesh {
        bool _loaded = false;
        std::vector<std::shared_ptr<Light>> lights;
        std::unordered_map<int, const Light *> lightMap;

      public:
        const Vertex *vertex_buffer;
        const int *index_buffer;
        const int *groups;
        std::shared_ptr<BinaryGeometry> geometry;
        [[refl]] std::vector<MaterialSlot> materials;
        [[refl]] fs::path file;
        AKR_IMPLS(Mesh)
        const MaterialSlot &get_material_slot(int group) const override;
        const Vertex *get_vertex_buffer() const override;
        const int *get_index_buffer() const override;
        size_t triangle_count() const override;
        size_t vertex_count() const override;
        int get_primitive_group(int idx) const override;
        bool load_path(const char *path) override;
        void save_path(const char *path) override;
        void commit() override;
        std::vector<std::shared_ptr<Light>> get_mesh_lights() const override;
        const Light *get_light(int primId) const override;
        std::vector<MaterialSlot> &GetMaterials() override { return materials; }
    };

    const MaterialSlot &BinaryMesh::get_material_slot(int group) const { return materials[group]; }
    const Vertex *BinaryMesh::get_vertex_buffer() const { return vertex_buffer; }
    const int *BinaryMesh::get_index_buffer() const { return index_buffer; }
    size_t BinaryMesh::triangle_count() const { return geometry->index_buffer.size() / 3; }
    size_t BinaryMesh::vertex_count() const { return geometry->vertex_buffer.size(); }
    int BinaryMesh::get_primitive_group(int idx) const { return groups[idx]; }

    bool BinaryMesh::load_path(const char *path) {
        if (auto _ = resource_manager()->load_path<BinaryGeometry>(path)) {
            geometry = *_;
            vertex_buffer = geometry->vertex_buffer.data();
            index_buffer = geometry->index_buffer.data();
            groups = geometry->groups.data();
            return true;
        }
        return false;
    }
    void BinaryMesh::save_path(const char *path) { geometry->save(fs::path(path)); }
    void BinaryMesh::commit() {
        if (_loaded) {
            return;
        }
        for (auto &mat : materials) {
            if (mat.emission.color) {
                mat.emission.color->commit();
            }
            if (mat.emission.strength) {
                mat.emission.strength->commit();
            }
            if(mat.material)
                mat.material->commit();
        }
        load_path(file.string().c_str());
        for (uint32_t id = 0; id < triangle_count(); id++) {
            int group = get_primitive_group(id);
            const auto &mat = get_material_slot(group);
            if (!mat.marked_as_light) {
                continue;
            }

            if (!mat.emission.color || !mat.emission.strength) {
                info("Mesh [{}] prim: [{}] of group: [{}] marked as light but don't have texture\n");
                continue;
            }

            lights.emplace_back(CreateAreaLight(*this, id));
            lightMap[id] = lights.back().get();
        }
    }
    std::vector<std::shared_ptr<Light>> BinaryMesh::get_mesh_lights() const { return lights; }
    const Light *BinaryMesh::get_light(int primId) const {
        auto it = lightMap.find(primId);
        if (it == lightMap.end()) {
            return nullptr;
        }
        return it->second;
    }
    std::shared_ptr<Mesh> create_binary_mesh(const fs::path &file, std::vector<Vertex> vertex, std::vector<int> index,
                                             std::vector<int> groups, std::vector<MaterialSlot> materials) {
        auto mesh = std::make_shared<BinaryMesh>();
        mesh->file = file;
        std::shared_ptr<BinaryGeometry> geometry = std::make_shared<BinaryGeometry>();
        geometry->vertex_buffer = std::move(vertex);
        geometry->index_buffer = std::move(index);
        geometry->groups = std::move(groups);
        mesh->geometry = geometry;
        mesh->materials = std::move(materials);
        return mesh;
    }
#include "generated/BinaryMesh.hpp"
    AKR_EXPORT_PLUGIN(p) {}
} // namespace akari