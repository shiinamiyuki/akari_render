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

#ifndef AKARIRENDER_PLUGIN_H
#define AKARIRENDER_PLUGIN_H

#include <Akari/Core/Akari.h>
#include <Akari/Core/Component.h>
#include <memory>

namespace Akari {

    typedef void (*SharedLibraryFunc)();
    class AKR_EXPORT SharedLibraryLoader {
        void *handle = nullptr;
        void Load(const char *path);

      public:
        explicit SharedLibraryLoader(const char *path) { Load(path); }
        SharedLibraryFunc GetFuncPointer(const char *name);

        ~SharedLibraryLoader();
    };

    class AKR_EXPORT IPlugin {
      public:
        virtual TypeInfo *GetTypeInfo() = 0;
        virtual const char * GetInterface() = 0;
    };

    class AKR_EXPORT IPluginManager {
      public:
        virtual void SetPluginPath(const char * path) = 0;
        virtual bool LoadPath(const char *path) = 0;
        virtual IPlugin * LoadPlugin(const char *name) = 0;
        virtual void ForeachPlugin(const std::function<void(IPlugin *)> & func) = 0;
    };

    AKR_EXPORT IPluginManager *GetPluginManager();

} // namespace Akari
#endif // AKARIRENDER_PLUGIN_H
