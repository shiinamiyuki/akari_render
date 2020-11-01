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

#include <akari/render/scenegraph.h>
#include <akari/render/texture.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/core/color.h>
#include <akari/render/common.h>
#include <akari/core/mesh.h>
namespace akari::render {
    class AkariMesh final : public MeshNode {
      public:
        std::string path;
        AkariMesh() = default;
        AkariMesh(std::string path) : path(path) {}
        std::shared_ptr<Mesh> mesh;
        std::vector<std::shared_ptr<MaterialNode>> materials;
        void commit() override {
            auto exp = resource_manager()->load_path<BinaryGeometry>(path);
            if (exp) {
                mesh = exp.extract_value()->mesh();
            } else {
                auto err = exp.extract_error();
                error("error loading {}: {}", path, err.what());
                throw std::runtime_error("Error loading mesh");
            }
        }
        MeshInstance create_instance(Allocator<> allocator) override {
            commit();
            MeshInstance instance;
            instance.indices = BufferView(mesh->indices.data(), mesh->indices.size());
            instance.material_indices = BufferView(mesh->material_indices.data(), mesh->material_indices.size());
            instance.normals = BufferView(mesh->normals.data(), mesh->normals.size());
            instance.texcoords = BufferView(mesh->texcoords.data(), mesh->texcoords.size());
            instance.vertices = BufferView(mesh->vertices.data(), mesh->vertices.size());
            // AKR_ASSERT_THROW(mesh->material_indices.size() == materials.size());
            instance.materials.resize(materials.size());
            for (size_t i = 0; i < materials.size(); i++) {
                instance.materials[i] = (materials[i]->create_material(allocator));
            }
            return instance;
        }
        void set_material(uint32_t index, const std::shared_ptr<MaterialNode> &mat) {
            AKR_ASSERT(mat);
            if (index >= materials.size()) {
                materials.resize(index + 1);
            }
            materials[index] = mat;
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "path") {
                path = value.get<std::string>().value();
            } else if (field == "materials") {
                AKR_ASSERT_THROW(value.is_array());
                for (auto mat : value) {
                    auto m = dyn_cast<MaterialNode>(mat.object());
                    AKR_ASSERT(m);
                    materials.emplace_back(m);
                }
            }
        }
    };
    AKR_EXPORT_NODE(AkariMesh, AkariMesh)
} // namespace akari::render