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
#include <array>
#include <akari/core/math.h>
#include <akari/render/scenegraph.h>
#include <akari/render/material.h>
namespace akari::render {
    class Material;
    struct Triangle {
        astd::array<Vec3, 3> vertices;
        astd::array<Vec3, 3> normals;
        astd::array<vec2, 3> texcoords;
        Material material;
        AKR_XPU Vec3 p(const vec2 &uv) const { return lerp3(vertices[0], vertices[1], vertices[2], uv); }
        AKR_XPU Float area() const {
            return length(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) * 0.5f;
        }
        AKR_XPU Vec3 ng() const { return normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])); }
        AKR_XPU Vec3 ns(const vec2 &uv) const { return lerp3(normals[0], normals[1], normals[2], uv); }
        AKR_XPU vec2 texcoord(const vec2 &uv) const { return lerp3(texcoords[0], texcoords[1], texcoords[2], uv); }

        AKR_XPU astd::optional<astd::pair<Float, Vec2>> intersect(const Ray &ray) const {
            auto &v0 = vertices[0];
            auto &v1 = vertices[1];
            auto &v2 = vertices[2];
            Vec3 e1 = (v1 - v0);
            Vec3 e2 = (v2 - v0);
            Float a, f, u, v;
            auto h = cross(ray.d, e2);
            a = dot(e1, h);
            if (a > Float(-1e-6f) && a < Float(1e-6f))
                return astd::nullopt;
            f = 1.0f / a;
            auto s = ray.o - v0;
            u = f * dot(s, h);
            if (u < 0.0 || u > 1.0)
                return astd::nullopt;
            auto q = cross(s, e1);
            v = f * dot(ray.d, q);
            if (v < 0.0 || u + v > 1.0)
                return astd::nullopt;
            Float t = f * dot(e2, q);
            if (t > ray.tmin && t < ray.tmax) {
                return astd::make_pair(t, Vec2(u, v));
            } else {
                return astd::nullopt;
            }
        }
    };
    class ShapeNode : public SceneGraphNode {
      public:
    };

} // namespace akari::render