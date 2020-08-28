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

#include <akari/common/taggedpointer.h>
#include <akari/common/color.h>
#include "gtest/gtest.h"

using namespace akari;
TEST(TestTaggedPointer, Type) {
    using P = TaggedPointer<int, float, double>;
    int *i = new int(123);
    P p(i);
    ASSERT_TRUE(p.isa<int>());
    ASSERT_FALSE(p.isa<float>());
    ASSERT_EQ(p.get_ptr(), i);
    ASSERT_EQ(*p.cast<int>(), *i);
    float *f = new float(1.23);
    p = f;
    ASSERT_FALSE(p.isa<int>());
    ASSERT_TRUE(p.isa<float>());
    ASSERT_EQ(p.get_ptr(), f);
    ASSERT_EQ(*p.cast<float>(), *f);
    delete i;
    delete f;
}

TEST(TestTaggedPointer, Dispatch) {
    using P = TaggedPointer<int, float, double>;
    float *i = new float(123);
    P p(i);
    p.dispatch([=](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, float>) {
            ASSERT_EQ(arg, *i);
        } else {
            ASSERT_TRUE(false);
        }
    });
}