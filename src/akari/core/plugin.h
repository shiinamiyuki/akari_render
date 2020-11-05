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
#pragma once
#include <unordered_map>
#include <list>
#include <memory>
#include <akari/core/platform.h>
#include <akari/core/akari.h>
#include <akari/core/logger.h>

namespace akari {
    class AKR_EXPORT SharedLibrary {
        void *handle = nullptr;
        void load(const char *path);

      public:
        explicit SharedLibrary(const char *path) { load(path); }
        void *function(const char *name);

        ~SharedLibrary();
    };

    struct CompilerInfo {
        const char *name;
        const char *version;
    };
    struct BuildInfo {
        const char *date;
        const char *time;
    };
    struct PluginDescriptor {
        const char *name;
        CompilerInfo compiler_info;
        BuildInfo build_info;
    };
    template <typename T>
    class PluginInterface {
      public:
        virtual const PluginDescriptor *descriptor() const = 0;
        virtual std::shared_ptr<T> make_shared() const = 0;
        virtual std::unique_ptr<T> make_unique() const = 0;
        virtual ~PluginInterface() = default;
    };
    template <typename T>
    class AKR_EXPORT PluginManager {
        fs::path prefix = core_globals()->program_path.parent_path() / "plugins";
        std::unordered_map<std::string, const PluginInterface<T> *> plugins;
        std::list<std::shared_ptr<SharedLibrary>> sharedLibraries;

      public:
        bool load_path(const std::string &path) {
            info("loading {}", path);
            auto lib = std::make_shared<SharedLibrary>(path.c_str());
            if (!lib) {
                error("cannot load {}", path);
                return false;
            }
            auto p = (PluginInterface<T> * (*)()) lib->function("akari_plugin_onload");
            if (!p) {
                error("cannot resolve symbol {} in {}\n", "akari_plugin_onload", path);
                return false;
            }
            const PluginInterface<T> *info = p();
            plugins.emplace(info->descriptor()->name, info);
            sharedLibraries.emplace_back(lib);
            return true;
        }
        const PluginInterface<T> *try_load_plugin(const std::string &name) { return load_plugin_impl(name, false); }
        const PluginInterface<T> *load_plugin(const std::string &name) { return load_plugin_impl(name, true); }

      private:
        const PluginInterface<T> *load_plugin_impl(const std::string &name, bool required) {
            auto it = plugins.find(name);
            if (it != plugins.end()) {
                return it->second;
            }
            auto path = prefix / fs::path(name);
#ifdef _WIN32
            path = path.concat(".dll");
#else
            path = path.concat(".so");
#endif
            if (load_path(path.string().c_str())) {
                it = plugins.find(name);
                if (it != plugins.end()) {
                    return it->second;
                }
                if (required)
                    fatal("Unknown plugin: `{}`\n", name);
                return nullptr;
            }
            return nullptr;
        }
    };
#define AKR_STRINGFY(x) #x
    static inline void get_compiler_info(CompilerInfo *compiler_info) {
#ifdef __GNUG__
        compiler_info->name = "g++";
        compiler_info->version = __VERSION__;
#else
        compiler_info->name = "MSVC";
        compiler_info->version = AKR_STRINGFY(_MSC_VER);
#endif
    }
    static inline void get_build_info(BuildInfo *build_info) {
        build_info->date = __DATE__;
        build_info->time = __TIME__;
    }

    template <typename Base, typename T>
    class PluginInterfaceImpl : public PluginInterface<Base> {
        PluginDescriptor desc;

      public:
        static_assert(std::is_final_v<T>);
        PluginInterfaceImpl(const char *name) {
            desc.name = name;
            get_compiler_info(&desc.compiler_info);
            get_build_info(&desc.build_info);
        }
        const PluginDescriptor *descriptor() const override { return &desc; }
        std::shared_ptr<Base> make_shared() const override { return std::make_shared<T>(); }
        std::unique_ptr<Base> make_unique() const override { return std::make_unique<T>(); }
    };
#define AKR_EXPORT_PLUGIN(PLUGIN, CLASS, INTERFACE)                                                                    \
    extern "C" AKR_EXPORT const PluginInterface<INTERFACE> *akari_plugin_onload() {                                    \
        static PluginInterfaceImpl<INTERFACE, CLASS> pi(#PLUGIN);                                                      \
        return &pi;                                                                                                    \
    }
} // namespace akari