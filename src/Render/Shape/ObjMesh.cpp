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

#include <Akari/Core/Plugin.h>
#include <Akari/Render/Geometry.hpp>
#include <Akari/Render/Material.h>
#include <tiny_obj_loader.h>
namespace Akari {
    class ObjMesh : public Mesh {
        std::vector<std::shared_ptr<Material>> materials;
        std::string path;

      public:
        AKR_SER(materials, path)
        AKR_DECL_COMP(ObjMesh, "ObjMesh")
        std::shared_ptr<MaterialSlot> GetMaterial(int group) const override { return nullptr; }
        std::shared_ptr<TriangleMesh> BuildMesh() const override { return std::shared_ptr<TriangleMesh>(); }
        IndexBuffer GetIndexBuffer() const override { return nullptr; }
        NormalBuffer GetNormalBuffer() const override { return nullptr; }
        VertexBuffer GetVertexBuffer() const override { return nullptr; }
        TexCoordBuffer GetTexCoordBuffer() const override { return nullptr; }
        size_t GetVertexCount() const override { return 0; }
        int GetPrimitiveGroup(int idx) const override { return 0; }
        bool Load(const char *_path) const override { return false; }
    };
    AKR_EXPORT_COMP(ObjMesh, "Mesh")
} // namespace Akari