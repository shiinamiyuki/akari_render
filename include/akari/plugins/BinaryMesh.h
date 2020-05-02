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

#ifndef AKARIRENDER_BINARYMESH_H
#define AKARIRENDER_BINARYMESH_H

#include <akari/Core/Plugin.h>
#include <akari/Render/Material.h>
#include <akari/Render/Mesh.h>

namespace akari {
    class AKR_EXPORT BinaryMesh : public Mesh {
        bool _loaded = false;
        std::vector<std::shared_ptr<Light>> lights;
        std::unordered_map<int, const Light *> lightMap;

      public:
        std::vector<Vertex> vertexBuffer;
        std::vector<int> indexBuffer;
        std::vector<int> groups;
        std::vector<MaterialSlot> materials;
        fs::path file;
        AKR_SER(materials, file);
        AKR_DECL_COMP(BinaryMesh, "BinaryMesh")
        const MaterialSlot &get_material_slot(int group) const override;
        const Vertex *get_vertex_buffer() const override;
        const int *get_index_buffer() const override;
        size_t triangle_count() const override;
        size_t vertex_count() const override;
        int get_primitive_group(int idx) const override;
        bool load(const char *path) override;
        void save(const char *path);
        void commit() override;
        std::vector<std::shared_ptr<Light>> get_mesh_lights() const override;
        const Light *get_light(int primId) const override;
        std::vector<MaterialSlot> &GetMaterials() override { return materials; }
    };
} // namespace akari

#endif // AKARIRENDER_BINARYMESH_H
