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

#include <akari/core/plugin.h>

#ifdef _WIN32

#    include <Windows.h>

namespace akari {
    void SharedLibrary::load(const char *path) { handle = LoadLibraryA(path); }

    void *SharedLibrary::function(const char *name) {
        auto func = (void *)GetProcAddress((HMODULE)handle, name);
        return func;
    }

    SharedLibrary::~SharedLibrary() {
        if (handle)
            FreeLibrary((HMODULE)handle);
    }

} // namespace akari
#else
#    include <dlfcn.h>
namespace akari {
    void SharedLibrary::load(const char *path) { handle = dlopen(path, RTLD_LAZY); }

    void *SharedLibrary::function(const char *name) { return (void *)dlsym(handle, name); }

    SharedLibrary::~SharedLibraryLoader() {
        if (handle)
            dlclose(handle);
    }
} // namespace akari

#endif

namespace akari {
    
}
