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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <akari/common/math.hpp>



int main() {
    AKR_USING_TYPES(akari::CoreAliases<float>, Point2f)
    using Vector3f = akari::Vector<float, 3>;
    using Vector4f = akari::Vector<float, 4>;
    using Point3f = akari::Point<float, 3>;
    using Normal3f = akari::Normal<float, 3>;
    auto v = Vector3f(1.0);
    auto p = Point3f(0);
    auto p2 = v + p;
    v = (sin(v));
    v = pow(v, Vector3f(2.0));
    using Matrix4f = akari::Matrix<float, 4>;
    Matrix4f m(2.0);
    m = m + m;
    m(2, 3) = 10.0;
    using namespace akari;
    using Frame3f = Frame<Vector3f>;
    Frame3f f(Normal3f(normalize(v)));
    Transform<float> transform(m);
    transform *v;
    // static_assert(sizeof(Array<Array<float, 4>, 4>) == sizeof(float[4][4]));
    auto minv = m.inverse();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", m(i, j));
        }
        putchar('\n');
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", minv(i, j));
        }
        putchar('\n');
    }
    auto id = m * minv;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", id(i, j));
        }
        putchar('\n');
    }
    using Bounds3f = akari::BoundingBox<Point3f>;
    Bounds3f bb;
    bb.surface_area();
}