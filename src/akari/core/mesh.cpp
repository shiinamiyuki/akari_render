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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>
#include <akari/core/mesh.h>
#include <akari/core/logger.h>
namespace akari {
    const char *AKR_MESH_MAGIC = "AKARI_BINARY_MESH";
    Expected<bool> BinaryGeometry::save(const fs::path &path) const {
        info("save to {}", path.string());
        std::ofstream out(path, std::ios::binary | std::ios::out);
        out.write(AKR_MESH_MAGIC, strlen(AKR_MESH_MAGIC));
        size_t vertexCount = _mesh->vertices.size() / 3;
        size_t triangleCount = _mesh->material_indices.size();
        AKR_ASSERT(_mesh->vertices.size() == vertexCount * 3);
        AKR_ASSERT(_mesh->normals.size() == triangleCount * 9);
        AKR_ASSERT(_mesh->texcoords.size() == triangleCount * 6);
        AKR_ASSERT(_mesh->indices.size() == triangleCount * 3);
        out.write(reinterpret_cast<const char *>(&vertexCount), sizeof(size_t));
        out.write(reinterpret_cast<const char *>(&triangleCount), sizeof(size_t));
        out.write(reinterpret_cast<const char *>(_mesh->vertices.data()), sizeof(float) * _mesh->vertices.size());
        out.write(reinterpret_cast<const char *>(_mesh->normals.data()), sizeof(float) * _mesh->normals.size());
        out.write(reinterpret_cast<const char *>(_mesh->texcoords.data()), sizeof(float) * _mesh->texcoords.size());
        out.write(reinterpret_cast<const char *>(_mesh->indices.data()), sizeof(int) * _mesh->indices.size());
        out.write(reinterpret_cast<const char *>(_mesh->material_indices.data()), sizeof(int) * triangleCount);
        out.write(AKR_MESH_MAGIC, strlen(AKR_MESH_MAGIC));
        return true;
    }
    Expected<bool> BinaryGeometry::load(const fs::path &path) {
        info("Loading {}", path.string());
        std::ifstream in(path, std::ios::binary | std::ios::in);
        if (!in.is_open()) {
            error("Cannot open file");
            return Error("No file");
        }
        char buffer[128] = {0};
        in.read(buffer, strlen(AKR_MESH_MAGIC));
        if (strcmp(buffer, AKR_MESH_MAGIC) != 0) {
            error("Failed to load mesh: invalid format; expected '{}' but found '{}'", AKR_MESH_MAGIC, buffer);
            return Error("Invalid format");
        }
        _mesh = std::make_shared<Mesh>(default_resource());
        size_t vertexCount;
        size_t triangleCount;
        in.read(reinterpret_cast<char *>(&vertexCount), sizeof(size_t));
        in.read(reinterpret_cast<char *>(&triangleCount), sizeof(size_t));
        _mesh->vertices.resize(vertexCount * 3);
        AKR_ASSERT(_mesh->vertices.data() != nullptr);
        in.read(reinterpret_cast<char *>(_mesh->vertices.data()), sizeof(float) * vertexCount * 3);
        _mesh->normals.resize(triangleCount * 9);
        in.read(reinterpret_cast<char *>(_mesh->normals.data()), sizeof(float) * triangleCount * 9);
        _mesh->texcoords.resize(triangleCount * 6);
        in.read(reinterpret_cast<char *>(_mesh->texcoords.data()), sizeof(float) * triangleCount * 6);
        _mesh->indices.resize(triangleCount * 3);
        in.read(reinterpret_cast<char *>(_mesh->indices.data()), sizeof(int) * triangleCount * 3);
        _mesh->material_indices.resize(triangleCount);
        in.read(reinterpret_cast<char *>(_mesh->material_indices.data()), sizeof(int) * triangleCount);
        memset(buffer, 0, sizeof(buffer));
        in.read(buffer, strlen(AKR_MESH_MAGIC));
        if (strcmp(buffer, AKR_MESH_MAGIC) != 0) {
            error("Failed to load mesh: invalid format; expected '{}' but found '{}'", AKR_MESH_MAGIC, buffer);
            return Error("Invalid format");
        }
        info("Loaded {} triangles", triangleCount);
        return true;
    }
} // namespace akari