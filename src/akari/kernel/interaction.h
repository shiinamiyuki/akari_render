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
#include <akari/kernel/material.h>
#include <akari/kernel/shape.h>
namespace akari {
    AKR_VARIANT class BSDF;
    AKR_VARIANT struct Intersection;
    AKR_VARIANT struct SurfaceInteraction {
        AKR_IMPORT_TYPES()
        Point3f p;
        Triangle<C> triangle;
        BSDF<C> bsdf;
        Normal3f ng, ns;
        Point2f texcoords;

        SurfaceInteraction(const Intersection<C> &isct, const Triangle<C> &triangle)
            : p(isct.p), triangle(triangle), ng(isct.ng), ns(triangle.ns(isct.uv)),
              texcoords(triangle.texcoord(isct.uv)) {}
    };
} // namespace akari