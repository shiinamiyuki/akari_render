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

#include <akari/core/buffer.h>
#include <akari/core/math.h>
#include "gtest/gtest.h"
using namespace akari;

TEST(TestBuffer, Basic) {
    astd::pmr::vector<ivec3> vec;
    ASSERT_EQ(vec.size(), 0);
    ASSERT_TRUE(vec.data() == nullptr);
    vec.resize(32);
    ASSERT_TRUE(vec.data() != nullptr);
    ASSERT_TRUE(vec.capacity() != 0);
    ASSERT_EQ(vec.size(), 32);
    vec.resize(1024);
    ASSERT_TRUE(vec.data() != nullptr);
    ASSERT_TRUE(vec.capacity() != 0);
    ASSERT_EQ(vec.size(), 1024);
    for (auto &i : vec) {
        i = ivec3(77, 88, 99);
    }
    ASSERT_TRUE(all(vec[64] == vec[1023]));
}

TEST(TestBuffer, Resize) {
     astd::pmr::vector<float> vec;
    vec.resize(216);
    ASSERT_TRUE(vec.data() != nullptr);
    ASSERT_TRUE(vec.capacity() != 0);
    ASSERT_EQ(vec.size(), 216);
}
TEST(TestBuffer, PushBack) {
     astd::pmr::vector<int> vec;
    vec.resize(64);
    ASSERT_EQ(vec.size(), 64);
    ASSERT_GE(vec.capacity(), 64);
    for (int i = 0; i < 64; i++) {
        ASSERT_EQ(vec[i], 0);
    }
    vec.push_back(123);
    ASSERT_EQ(vec.size(), 65);
    for (int i = 0; i < 64; i++) {
        ASSERT_EQ(vec[i], 0);
    }
    ASSERT_EQ(vec[64], 123);
}