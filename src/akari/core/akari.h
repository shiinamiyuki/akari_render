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

#ifndef AKARIRENDER_AKARI_H
#define AKARIRENDER_AKARI_H
#include <akari/core/platform.h>
#include <akari/core/diagnostic.h>
#ifdef __GNUC__
#    if __GNUC__ >= 8

#        include <filesystem>
namespace akari {
    namespace fs = std::filesystem;
}
#    else
#        include <experimental/filesystem>
namespace akari {
    namespace fs = std::experimental::filesystem;
}
#    endif
#else
#    include <filesystem>
namespace akari {
    namespace fs = std::filesystem;
}

#endif

#include <functional>
#include <string_view>
#include <akari/core/panic.h>

namespace akari {

    struct CurrentPathGuard {
        fs::path _cur;
        CurrentPathGuard() : _cur(fs::current_path()) {}
        ~CurrentPathGuard() { fs::current_path(_cur); }
    };
    struct NonCopyable {
        NonCopyable() = default;
        NonCopyable(const NonCopyable &) = delete;
        NonCopyable &operator=(const NonCopyable &) = delete;
    };

    template <typename U, typename T>
    std::shared_ptr<U> dyn_cast(const std::shared_ptr<T> &p) {
        return std::dynamic_pointer_cast<U>(p);
    }

    template <typename T>
    struct ScopedAssignment {
        ScopedAssignment(T *p) : ptr(p), backup(*p) {}
        ~ScopedAssignment() { *ptr = backup; }

      private:
        T *ptr;
        T backup;
    };

    template <typename F>
    struct AtScopeExit {
        F f;
        AtScopeExit(F &&f) : f(f) {}
        ~AtScopeExit() { f(); }
    };
    template <typename F>
    AtScopeExit(F &&f) -> AtScopeExit<F>;

    struct CoreGlobals {
        fs::path program_path;
    };
    AKR_EXPORT CoreGlobals *core_globals();

   
} // namespace akari
#endif // AKARIRENDER_AKARI_H
