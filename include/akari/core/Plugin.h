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

#include <akari/core/akari.h>
#include <akari/core/serialize.hpp>
#include <akari/core/detail/serialize-impl.hpp>
#include <memory>
#include <mutex>

namespace akari {

    typedef void (*SharedLibraryFunc)();
    class AKR_EXPORT SharedLibraryLoader {
        void *handle = nullptr;
        void load(const char *path);

      public:
        explicit SharedLibraryLoader(const char *path) { load(path); }
        SharedLibraryFunc load_function(const char *name);

        ~SharedLibraryLoader();
    };
    struct CompilerInfo {
        const char *name;
        const char *version;
    };
    struct BuildInfo {
        const char *date;
        const char *time;
    };
    struct Plugin {
        const char *name{};
        CompilerInfo compiler_info{};
        BuildInfo build_info{};
    };

    class AKR_EXPORT PluginManager {
      public:
        virtual void set_plugin_path(const char *path) = 0;
        virtual bool load_path(const char *path) = 0;
        virtual bool load_plugin(const char *name) = 0;
    };

#define AKR_EXPORT_PLUGIN(_name, _p)                                                                                   \
    void _AkariPluginOnLoad(Plugin &);                                                                                 \
    extern "C" AKR_EXPORT Plugin *akari_plugin_onload() {                                                              \
        static Plugin plugin;                                                                                          \
        static std::once_flag flag;                                                                                    \
        std::call_once(flag, [&]() {                                                                                   \
            plugin.name = #_name;                                                                                      \
            get_compiler_info(&plugin.compiler_info);                                                                  \
            get_build_info(&plugin.build_info);                                                                        \
            _AkariPluginOnLoad(plugin);                                                                                \
        });                                                                                                            \
        return &plugin;                                                                                                \
    }                                                                                                                  \
    void _AkariPluginOnLoad(Plugin &_p)

    AKR_EXPORT PluginManager *get_plugin_manager();

    static inline void get_compiler_info(CompilerInfo *compiler_info) {
#ifdef __GNUG__
        compiler_info->name = "g++";
        compiler_info->version = __VERSION__;
#else
        compiler_info->name = "unknown";
        compiler_info->version = "unknown";
#endif
    }
    static inline void get_build_info(BuildInfo *build_info) {
        build_info->date = __DATE__;
        build_info->time = __TIME__;
    }

} // namespace akari
#endif // AKARIRENDER_PLUGIN_H
