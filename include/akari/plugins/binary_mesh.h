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

#include <akari/core/plugin.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/core/resource.h>

namespace akari {
    AKR_EXPORT std::shared_ptr<Mesh> create_binary_mesh(const fs::path& file, std::vector<Vertex> vertex,
                                             std::vector<int> index, std::vector<int> groups,
                                             std::vector<MaterialSlot> materials);

    struct BinaryGeometry : Resource{
        std::vector<Vertex> vertex_buffer;
        std::vector<int> index_buffer;
        std::vector<int> groups;
        Expected<bool> load(const fs::path &)override;
        Expected<bool> save(const fs::path &)const;
    };
} // namespace akari

#endif // AKARIRENDER_BINARYMESH_H
