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

#include <akari/kernel/cameras/camera.h>

namespace akari {
    AKR_VARIANT void PerspectiveCamera<Float, Spectrum>::preprocess() {
        Transform3f m;
        m = Transform3f::scale(Vector3f(1.0f / resolution.x(), 1.0f / resolution.y(), 1)) * m;
        m = Transform3f::scale(Vector3f(2, 2, 1)) * m;
        m = Transform3f::translate(Point3f(-1, -1, 0)) * m;
        m = Transform3f::scale(Vector3f(1, -1, 1)) * m;
        auto s = std::atan(fov / 2);
        if (resolution.x() > resolution.y()) {
            m = Transform3f::scale(Vector3f(s, s * Float(resolution.y()) / resolution.x(), 1)) * m;
        } else {
            m = Transform3f::scale(Vector3f(s * Float(resolution.x()) / resolution.y(), s, 1)) * m;
        }
        r2w = m;
    }
    AKR_RENDER_CLASS(PerspectiveCamera)
} // namespace akari