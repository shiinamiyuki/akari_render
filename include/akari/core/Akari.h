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
#pragma warning(push, 4)
#pragma warning(disable : 4100)
#pragma warning(disable : 4244)
#pragma warning(disable : 4146)
#pragma warning(disable : 4305)
#pragma warning(disable : 5030)
#else
#pragma GCC diagnostic error "-Wall"
#pragma clang diagnostic error "-Wall"
#pragma GCC diagnostic ignored "-Wc++11-compat"
#pragma clang diagnostic ignored "-Wc++11-compat"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wattributes"
#pragma clang diagnostic ignored "-Wattributes"
#endif

#ifdef __GNUC__
#if __GNUC__ >= 8

#include <filesystem>
namespace akari {
    namespace fs = std::filesystem;
}
#else
#include <experimental/filesystem>
namespace akari {
    namespace fs = std::experimental::filesystem;
}
#endif
#else
#include <filesystem>
namespace akari {
    namespace fs = std::filesystem;
}

#endif

#include <functional>
#include <string_view>

namespace akari {
    [[noreturn]] inline void panic(const char *file, int line, const char *msg) {
        fprintf(stderr, "PANIC at %s:%d: %s\n", file, line, msg);
        abort();
    }

    struct CurrentPathGuard {
        fs::path _cur;
        CurrentPathGuard() : _cur(fs::current_path()) {}
        ~CurrentPathGuard() { fs::current_path(_cur); }
    };
#define AKARI_NON_COPYABLE(CLASS)                                                                                      \
    CLASS(const CLASS &) = delete;                                                                                     \
    CLASS &operator=(const CLASS &) = delete;
#define AKR_PANIC(msg) panic(__FILE__, __LINE__, msg)
#define AKARI_CHECK(expr)                                                                                              \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, #expr " not satisfied at %s:%d\n", __FILE__, __LINE__);                                    \
        }                                                                                                              \
    } while (0)
#define AKR_ASSERT(expr)                                                                                             \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            AKR_PANIC(#expr " not satisfied");                                                                       \
        }                                                                                                              \
    } while (0)

//    using Float = float;
} // namespace akari
#endif // AKARIRENDER_AKARI_H
