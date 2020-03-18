// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#ifndef AKARIRENDER_AKARI_H
#define AKARIRENDER_AKARI_H

#ifdef _MSC_VER
#else
#pragma GCC diagnostic error "-Wall"
#pragma clang diagnostic error "-Wall"
#endif

#include <filesystem>
#include <functional>
#include <string_view>

namespace Akari {
    namespace fs = std::filesystem;
    [[noreturn]] inline void panic(const char *msg) {
        fprintf(stderr, "%s\n", msg);
        abort();
    }

    struct CurrentPathGuard{
        fs::path _cur;
        CurrentPathGuard():_cur(fs::current_path()){}
        ~CurrentPathGuard(){
            fs::current_path(_cur);
        }
    };
#define AKARI_NON_COPYABLE(CLASS)                                                                                      \
    CLASS(const CLASS &) = delete;                                                                                     \
    CLASS &operator=(const CLASS &) = delete;
#define _AKARI_STR(x)    #x
#define AKARI_STR(x)     _AKARI_STR(x)
#define AKARI_PANIC(msg) panic(__FILE__, __LINE__, msg)
#define AKARI_ASSERT(expr)                                                                                             \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            AKARI_PANIC(#expr " not satisfied at " __FILE__ ":" AKARI_STR(__LINE__));                                  \
        }                                                                                                              \
    } while (0)
} // namespace Akari
#endif // AKARIRENDER_AKARI_H
