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
#include <akari/kernel/material.h>
namespace akari {
    AKR_VARIANT struct Triangle {
        AKR_IMPORT_TYPES()
        Array<Point3f, 3> vertices;
        Array<Normal3f, 3> normals;
        Array<Point2f, 3> texcoords;
        Material<C> material;
        Normal3f ng() const { return Normal3f(cross(vertices[1] - vertices[0], vertices[2] - vertices[1])); }
        Normal3f ns(const Point2f &uv) const { return lerp3(normals[0], normals[1], normals[2], uv); }
        Point2f texcoord(const Point2f &uv) const { return lerp3(texcoords[0], texcoords[1], texcoords[2], uv); }
    };
} // namespace akari