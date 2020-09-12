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
#include <akari/common/bufferview.h>
#include <akari/common/fwd.h>
#include <akari/kernel/material.h>
#include <akari/kernel/shape.h>
namespace akari {
    AKR_VARIANT struct MeshInstance {
        BufferView<float> vertices, normals, texcoords;
        BufferView<int> indices;
        BufferView<int> material_indices;
        BufferView<const Material<C> *> materials;
    };

    template <typename C, typename Mesh>
    Triangle<C> get_triangle(const Mesh &mesh, int prim_id) {
        AKR_IMPORT_TYPES()
        Triangle<C> trig;
        for (int i = 0; i < 3; i++) {
            int idx = mesh.indices[3 * prim_id + i];
            trig.vertices[i] =
                Point3f(mesh.vertices[3 * idx + 0], mesh.vertices[3 * idx + 1], mesh.vertices[3 * idx + 2]);
            idx = 3 * prim_id + i;
            trig.normals[i] = Normal3f(mesh.normals[3 * idx + 0], mesh.normals[3 * idx + 1], mesh.normals[3 * idx + 2]);
            trig.texcoords[i] = Point2f(mesh.texcoords[2 * idx + 0], mesh.texcoords[2 * idx + 1]);
        }
        return trig;
    }
} // namespace akari