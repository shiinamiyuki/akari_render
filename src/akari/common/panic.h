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
#include <cstdio>
#include <cstdlib>
namespace akari {
    [[noreturn]] inline void panic(const char *file, int line, const char *msg) {
        fprintf(stderr, "PANIC at %s:%d: %s\n", file, line, msg);
        abort();
    }
#define AKR_PANIC(msg) panic(__FILE__, __LINE__, msg)
#define AKR_CHECK(expr)                                                                                                \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, #expr " not satisfied at %s:%d\n", __FILE__, __LINE__);                                    \
        }                                                                                                              \
    } while (0)
#define AKR_ASSERT(expr)                                                                                               \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            AKR_PANIC(#expr " not satisfied");                                                                         \
        }                                                                                                              \
    } while (0)
#define AKR_ASSERT_THROW(expr)                                                                                         \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            throw std::runtime_error(#expr " not satisfied");                                                          \
        }                                                                                                              \
    } while (0)
} // namespace akari