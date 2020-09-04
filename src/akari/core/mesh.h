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

#pragma once
#include <akari/common/math.h>
#include <akari/core/buffer.h>
#include <akari/core/resource.h>
#include <akari/common/variant.h>
namespace akari {
    
    struct Mesh {
        Buffer<float> vertices, normals, texcoords;
        Buffer<int> indices;
        Buffer<int> material_indices;
    };

    class AKR_EXPORT BinaryGeometry : public Resource {
        std::shared_ptr<Mesh> _mesh;

      public:
        BinaryGeometry() = default;
        BinaryGeometry(std::shared_ptr<Mesh> mesh) : _mesh(std::move(mesh)) {}
        Expected<bool> load(const fs::path &) override;
        const std::shared_ptr<Mesh> &mesh() const { return _mesh; }
        Expected<bool> save(const fs::path &path) const;
    };
} // namespace akari