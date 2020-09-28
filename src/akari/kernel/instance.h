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
#include <akari/common/buffer.h>
#include <akari/common/fwd.h>
#include <akari/kernel/material.h>
#include <akari/kernel/shape.h>
namespace akari {
    AKR_VARIANT struct MeshInstance {
        AKR_IMPORT_TYPES()
        BufferView<float> vertices, normals, texcoords;
        BufferView<int> indices;
        BufferView<int> material_indices;
        BufferView<const Material<C> *> materials;
        struct RayHit {
            Point2f uv;
            Point3f ng;
            Float t = Constants<Float>::Inf();
            int prim_id = -1;
            int mat_idx = -1;
        };
        AKR_XPU bool intersect(const Ray3f &ray, int prim_id, RayHit *hit) const {
            int idx0 = indices[3 * prim_id + 0];
            int idx1 = indices[3 * prim_id + 1];
            int idx2 = indices[3 * prim_id + 2];
            auto v0 = Point3f(vertices[3 * idx0 + 0], vertices[3 * idx0 + 1], vertices[3 * idx0 + 2]);
            auto v1 = Point3f(vertices[3 * idx1 + 0], vertices[3 * idx1 + 1], vertices[3 * idx1 + 2]);
            auto v2 = Point3f(vertices[3 * idx2 + 0], vertices[3 * idx2 + 1], vertices[3 * idx2 + 2]);
            Vector3f e1 = (v1 - v0);
            Vector3f e2 = (v2 - v0);
            Float a, f, u, v;
            auto h = cross(ray.d, e2);
            a = dot(e1, h);
            if (a > Float(-1e-6f) && a < Float(1e-6f))
                return false;
            f = 1.0f / a;
            auto s = ray.o - v0;
            u = f * dot(s, h);
            if (u < 0.0 || u > 1.0)
                return false;
            auto q = cross(s, e1);
            v = f * dot(ray.d, q);
            if (v < 0.0 || u + v > 1.0)
                return false;
            Float t = f * dot(e2, q);
            if (t > ray.tmin && t < ray.tmax) {
                if (hit) {
                    if (t < hit->t) {
                        hit->uv = Point2f(u, v);
                        hit->prim_id = prim_id;
                        hit->mat_idx = material_indices[prim_id];
                        hit->t = t;
                        return true;
                    }
                    return false;
                }
                return true;
            } else {
                return false;
            }
        }
    };

    template <typename C, typename Mesh>
    AKR_XPU Triangle<C> get_triangle(const Mesh &mesh, int prim_id) {
        AKR_IMPORT_TYPES()
        Triangle<C> trig;
        for (int i = 0; i < 3; i++) {
            int idx = mesh.indices[3 * prim_id + i];
            trig.vertices[i] =
                Point3f(mesh.vertices[3 * idx + 0], mesh.vertices[3 * idx + 1], mesh.vertices[3 * idx + 2]);
            int vidx = 3 * prim_id + i;
            trig.normals[i] =
                Normal3f(mesh.normals[3 * vidx + 0], mesh.normals[3 * vidx + 1], mesh.normals[3 * vidx + 2]);
            trig.texcoords[i] = Point2f(mesh.texcoords[2 * vidx + 0], mesh.texcoords[2 * vidx + 1]);
        }
        return trig;
    }
} // namespace akari