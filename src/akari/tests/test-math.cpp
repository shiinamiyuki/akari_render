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

#include <akari/common/math.h>
#include <akari/common/color.h>
#include "gtest/gtest.h"
using namespace akari;
TEST(TestMath, Color) {
    using Float = float;
    AKR_IMPORT_CORE_TYPES()
    Color3f L(0.02, 0.4, 0.8);
    auto x = linear_to_srgb(L);
    ASSERT_LT(x[0], 1.0f);
    ASSERT_LT(x[1], 1.0f);
    ASSERT_LT(x[2], 1.0f);
}
TEST(TestMath, Frame) {
    using Float = float;
    AKR_IMPORT_CORE_TYPES()
    Frame3f frame(normalize(Vector3f(0.2, 0.7, 0.4)));
    Vector3f w = normalize(Vector3f(0.6, 0.3, 0.2));
    auto u = frame.local_to_world(w);
    auto v = frame.world_to_local(u);
    ASSERT_FLOAT_EQ(w[0], v[0]);
    ASSERT_FLOAT_EQ(w[1], v[1]);
    ASSERT_FLOAT_EQ(w[2], v[2]);
}