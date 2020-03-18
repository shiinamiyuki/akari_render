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

#ifndef AKARIRENDER_GEOMETRY_HPP
#define AKARIRENDER_GEOMETRY_HPP

#include <Akari/Core/Component.h>
#include <Akari/Core/Math.h>


namespace Akari {
    struct Ray {
        vec3 o, d;
        float t_min, t_max;

        Ray() = default;

        Ray(const vec3 &o, const vec3 &d, float t_min, float t_max = std::numeric_limits<float>::infinity())
            : o(o), d(d), t_min(t_min), t_max(t_max) {}
    };

    struct TriangleMesh {
        struct Vertex {
            vec3 position;
            vec3 normal;
            vec2 texCoord;
        };
        std::vector<Vertex> vertices;
        std::vector<ivec3> indices;
    };

    struct Triangle {
        std::array<vec3, 3> v;
        std::array<vec2, 3> texCoords;
        std::array<vec3, 3> Ns;
        vec3 Ng;
    };
    struct MaterialSlot;
    class AKR_EXPORT Mesh : public Component {
      public:
        using IndexBuffer = const ivec3 *;
        using NormalBuffer = const vec3 *;
        using VertexBuffer = const vec3 *;
        using TexCoordBuffer = const vec2 *;
        struct RayHit {
            vec2 uv;
            vec3 Ng;
            Float t = Inf;
            int face = -1;
            int group = -1;
        };
        virtual std::shared_ptr<MaterialSlot> GetMaterial(int group) const = 0;
        virtual std::shared_ptr<TriangleMesh> BuildMesh() const = 0;
        [[nodiscard]] virtual IndexBuffer GetIndexBuffer() const = 0;
        [[nodiscard]] virtual NormalBuffer GetNormalBuffer() const = 0;
        [[nodiscard]] virtual VertexBuffer GetVertexBuffer() const = 0;
        [[nodiscard]] virtual TexCoordBuffer GetTexCoordBuffer() const = 0;
        [[nodiscard]] virtual size_t GetVertexCount() const = 0;
        [[nodiscard]] virtual int GetPrimitiveGroup(int idx) const = 0;
        virtual bool Load(const char *path) const = 0;
        bool Intersect(const Ray &ray, int idx, RayHit *hit) const {
            auto vertices = GetVertexBuffer();
            auto indices = GetIndexBuffer();
            auto normals = GetNormalBuffer();
            auto texCoords = GetNormalBuffer();
            auto v0 = vertices[indices[idx][0]];
            auto v1 = vertices[indices[idx][1]];
            auto v2 = vertices[indices[idx][2]];
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

    struct Intersection {
        Float t;
        Triangle triangle;
        int32_t meshId = -1;
        int32_t primId = -1;
        int32_t primGroup = -1;
        vec2 uv;
        vec3 Ng;
    };
    class BSDF;

    struct ShadingPoint {
        vec2 texCoords;
    };

    struct ScatteringEvent {
        vec3 wo;
        vec3 p;
        vec3 wi;
        ShadingPoint sp;
        BSDF *bsdf = nullptr;
    };


} // namespace Akari
#endif // AKARIRENDER_GEOMETRY_HPP
