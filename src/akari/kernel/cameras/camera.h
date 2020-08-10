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
#include <akari/common/variant.h>
#include <akari/common/math.h>
namespace akari {
    AKR_VARIANT class PerspectiveCamera {
      public:
        AKR_IMPORT_BASIC_RENDER_TYPES()
      private:
        Transform3f c2w, w2c, r2w;
        Point2i resolution;
        Float fov;
        void preprocess();

      public:
        PerspectiveCamera(const Point2i &resolution, const Transform3f &c2w, Float fov)
            : resolution(resolution), c2w(c2w), w2c(c2w.inverse()), fov(fov) {
            preprocess();
        }
    };
    AKR_VARIANT class Camera : Variant<PerspectiveCamera<Float, Spectrum>> {
      public:
        using Variant::Variant;
        AKR_IMPORT_BASIC_RENDER_TYPES()
    };
} // namespace akari