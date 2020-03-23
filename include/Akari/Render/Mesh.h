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

#ifndef AKARIRENDER_MESH_H
#define AKARIRENDER_MESH_H

#include <Akari/Render/Geometry.hpp>

namespace Akari {
    struct MaterialSlot;
    class Light;
    class AKR_EXPORT Mesh : public Component {
      public:
        struct RayHit {
            vec2 uv;
            vec3 Ng;
            Float t = Inf;
            int face = -1;
            int group = -1;
        };

        [[nodiscard]] virtual const Vertex *GetVertexBuffer() const = 0;
        [[nodiscard]] virtual const int *GetIndexBuffer() const = 0;
        [[nodiscard]] virtual size_t GetTriangleCount() const = 0;
        [[nodiscard]] virtual int GetPrimitiveGroup(int idx) const = 0;
        virtual bool Load(const char *path) = 0;
        virtual const MaterialSlot &GetMaterialSlot(int group) const = 0;
        bool Intersect(const Ray &ray, int idx, RayHit *hit) const {
            auto vertices = GetVertexBuffer();
            auto indices = GetIndexBuffer();
            auto v0 = vertices[indices[idx * 3 + 0]].pos;
            auto v1 = vertices[indices[idx * 3 + 1]].pos;
            auto v2 = vertices[indices[idx * 3 + 2]].pos;
            vec3 e1 = (v1 - v0);
            vec3 e2 = (v2 - v0);
            auto Ng = normalize(cross(e1, e2));
            float a, f, u, v;
            auto h = cross(ray.d, e2);
            a = dot(e1, h);
            if (a > -1e-6f && a < 1e-6f)
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
            float t = f * dot(e2, q);
            if (t > ray.t_min && t < ray.t_max) {
                if (hit) {
                    if (t < hit->t) {
                        hit->Ng = Ng;
                        hit->uv = vec2(u, v);
                        hit->face = idx;
                        hit->group = GetPrimitiveGroup(idx);
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
        void GetTriangle(uint32_t primId, Triangle *triangle) const {
            auto vertices = GetVertexBuffer();
            auto indices = GetIndexBuffer();
            auto v0 = vertices[indices[primId * 3 + 0]].pos;
            auto v1 = vertices[indices[primId * 3 + 1]].pos;
            auto v2 = vertices[indices[primId * 3 + 2]].pos;
            vec3 e1 = (v1 - v0);
            vec3 e2 = (v2 - v0);
            auto Ng = normalize(cross(e1, e2));
            triangle->v[0] = v0;
            triangle->v[1] = v1;
            triangle->v[2] = v2;
            triangle->Ng = Ng;
            triangle->Ns[0] = vertices[indices[primId * 3 + 0]].Ns;
            triangle->Ns[1] = vertices[indices[primId * 3 + 1]].Ns;
            triangle->Ns[2] = vertices[indices[primId * 3 + 2]].Ns;
            triangle->texCoords[0] = vertices[indices[primId * 3 + 0]].texCoord;
            triangle->texCoords[1] = vertices[indices[primId * 3 + 1]].texCoord;
            triangle->texCoords[2] = vertices[indices[primId * 3 + 2]].texCoord;
        }

        [[nodiscard]] virtual std::vector<std::shared_ptr<Light>> GetMeshLights() const { return {}; }
    };
} // namespace Akari
#endif // AKARIRENDER_MESH_H
