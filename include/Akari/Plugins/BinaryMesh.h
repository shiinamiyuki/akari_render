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

#include <Akari/Core/Plugin.h>
#include <Akari/Render/Geometry.hpp>
#include <Akari/Render/Material.h>

namespace Akari {
    class AKR_EXPORT BinaryMesh : public Mesh {
        std::vector<Vertex> vertexBuffer;
        std::vector<ivec3> indexBuffer;
        std::vector<int> groups;
        std::vector<MaterialSlot> materials;
        std::string file;

      public:
        AKR_SER(materials, file);
        AKR_DECL_COMP(BinaryMesh, "BinaryMesh")
        const MaterialSlot &GetMaterialSlot(int group) const;
        const Vertex *GetVertexBuffer() const override;
        const ivec3 *GetIndexBuffer() const override;
        size_t GetTriangleCount() const override;
        int GetPrimitiveGroup(int idx) const override;
        bool Load(const char *path) override;
    };
} // namespace Akari

#endif // AKARIRENDER_BINARYMESH_H
