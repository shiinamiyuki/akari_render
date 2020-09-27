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

#include <akari/core/arena.h>
#include <akari/common/math.h>
#include <akari/common/variant.h>
#include "gtest/gtest.h"
using namespace akari;

TEST(TestArena, Alloc) {
    struct Foo {
        int arr[3] = {7, 8, 9};
    };
    auto arena = MemoryArena<>(astd::pmr::polymorphic_allocator<>());
    auto *p = arena.allocN<int>(1024, 0x7f);
    for (int i = 0; i < 1024; i++) {
        ASSERT_EQ(p[i], 0x7f);
    }
    arena.reset();
    auto v = *arena.alloc<Variant<Foo, int>>(Foo());
    ASSERT_TRUE(v.isa<Foo>());
    auto foo = *v.get<Foo>();
    ASSERT_EQ(foo.arr[0], 7);
    ASSERT_EQ(foo.arr[1], 8);
    ASSERT_EQ(foo.arr[2], 9);
}

#ifdef AKR_ENABLE_GPU
TEST(TestArena, GPUAlloc) {
    set_device_gpu();
    auto arena = MemoryArena<>(astd::pmr::polymorphic_allocator<>());
    auto *p = arena.allocN<int>(1024, 0x7f);
    for (int i = 0; i < 1024; i++) {
        ASSERT_EQ(p[i], 0x7f);
    }
    set_device_cpu();
}
#endif