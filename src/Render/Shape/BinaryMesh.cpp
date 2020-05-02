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

#include <Akari/Core/Logger.h>
#include <Akari/Core/Stream.h>
#include <Akari/Plugins/AreaLight.h>
#include <Akari/Plugins/BinaryMesh.h>
#include <fstream>
namespace Akari {
    const MaterialSlot &BinaryMesh::GetMaterialSlot(int group) const { return materials[group]; }
    const Vertex *BinaryMesh::GetVertexBuffer() const { return vertexBuffer.data(); }
    const int *BinaryMesh::GetIndexBuffer() const { return indexBuffer.data(); }
    size_t BinaryMesh::GetTriangleCount() const { return indexBuffer.size() / 3; }
    size_t BinaryMesh::GetVertexCount() const { return vertexBuffer.size(); }
    int BinaryMesh::GetPrimitiveGroup(int idx) const { return groups[idx]; }
    const char *AKR_MESH_MAGIC = "AKARI_BINARY_MESH";
    bool BinaryMesh::Load(const char *path) {
        Info("Loading {}\n", path);
        std::ifstream in(path, std::ios::binary | std::ios::in);
        char buffer[128] = {0};
        in.read(buffer, strlen(AKR_MESH_MAGIC));
        if (strcmp(buffer, AKR_MESH_MAGIC) != 0) {
            Error("Failed to load mesh: invalid format\n");
            return false;
        }
        size_t vertexCount;
        in.read(reinterpret_cast<char *>(&vertexCount), sizeof(size_t));
        vertexBuffer.resize(vertexCount);
        in.read(reinterpret_cast<char *>(vertexBuffer.data()), sizeof(Vertex) * vertexCount);
        indexBuffer.resize(vertexCount);
        in.read(reinterpret_cast<char *>(indexBuffer.data()), sizeof(int) * vertexCount);
        groups.resize(vertexCount / 3);
        in.read(reinterpret_cast<char *>(groups.data()), sizeof(int) * groups.size());
        memset(buffer, 0, sizeof(buffer));
        in.read(buffer, strlen(AKR_MESH_MAGIC));
        if (strcmp(buffer, AKR_MESH_MAGIC) != 0) {
            Error("Failed to load mesh: invalid format\n");
            return false;
        }
        _loaded = true;
        Info("Loaded {} triangles\n", groups.size());
        return true;
    }
    void BinaryMesh::Save(const char *path) {
        std::ofstream out(path, std::ios::binary | std::ios::out);
        out.write(AKR_MESH_MAGIC, strlen(AKR_MESH_MAGIC));
        size_t vertexCount = vertexBuffer.size();
        out.write(reinterpret_cast<char *>(&vertexCount), sizeof(size_t));
        out.write(reinterpret_cast<char *>(vertexBuffer.data()), sizeof(Vertex) * vertexCount);
        out.write(reinterpret_cast<char *>(indexBuffer.data()), sizeof(int) * vertexCount);
        out.write(reinterpret_cast<char *>(groups.data()), sizeof(int) * groups.size());
        out.write(AKR_MESH_MAGIC, strlen(AKR_MESH_MAGIC));
    }
    void BinaryMesh::Commit() {
        if (_loaded) {
            return;
        }
        Load(file.string().c_str());
        for (uint32_t id = 0; id < GetTriangleCount(); id++) {
            int group = GetPrimitiveGroup(id);
            const auto &mat = GetMaterialSlot(group);
            if (!mat.marked_as_light) {
                continue;
            }
            if (!mat.emission.color || !mat.emission.strength) {
                Info("Mesh [{}] prim: [{}] of group: [{}] marked as light but don't have texture\n");
                continue;
            }
            lights.emplace_back(CreateAreaLight(*this, id));
            lightMap[id] = lights.back().get();
        }
    }
    std::vector<std::shared_ptr<Light>> BinaryMesh::GetMeshLights() const { return lights; }
    const Light *BinaryMesh::GetLight(int primId) const {
        auto it = lightMap.find(primId);
        if (it == lightMap.end()) {
            return nullptr;
        }
        return it->second;
    }
    AKR_EXPORT_COMP(BinaryMesh, "Mesh");
} // namespace Akari