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

#include <akari/core/logger.h>
#include <akari/core/plugin.h>
#include <akari/core/reflect.hpp>
#ifdef _WIN32

#include <Windows.h>

namespace akari {
    void SharedLibraryLoader::load(const char *path) { handle = LoadLibraryA(path); }

    SharedLibraryFunc SharedLibraryLoader::load_function(const char *name) {
        auto func = (SharedLibraryFunc)GetProcAddress((HMODULE)handle, name);
        return func;
    }

    SharedLibraryLoader::~SharedLibraryLoader() {
        if (handle)
            FreeLibrary((HMODULE)handle);
    }

} // namespace akari
#else
#include <dlfcn.h>
namespace akari {
    void SharedLibraryLoader::load(const char *path) { handle = dlopen(path, RTLD_LAZY); }

    SharedLibraryFunc SharedLibraryLoader::load_function(const char *name) {
        return (SharedLibraryFunc)dlsym(handle, name);
    }

    SharedLibraryLoader::~SharedLibraryLoader() {
        if (handle)
            dlclose(handle);
    }
} // namespace akari

#endif

namespace akari {

    class PluginManagerImpl : public PluginManager {
        std::unordered_map<std::string, Plugin *> plugins;
        std::vector<std::unique_ptr<SharedLibraryLoader>> sharedLibraries;
        fs::path prefix = fs::absolute(fs::path("./"));

      public:
        void set_plugin_path(const char *path) override { prefix = path; }
        bool load_path(const char *path) override {
            info("Loading {}\n", path);
            auto lib = std::make_unique<SharedLibraryLoader>(path);
            if (!lib) {
                error("Cannot load {}\n", path);
                return false;
            }
            auto p = (Plugin* (*)())lib->load_function("akari_plugin_onload");
            if (!p) {
                error("Cannot resolve symbol {} in {}\n", "akari_plugin_onload", path);
                return false;
            }
            Plugin *info = p();
            plugins.emplace(info->name, info);
            sharedLibraries.emplace_back(std::move(lib));
            return true;
        }

        bool load_plugin(const char *name) override {
            auto it = plugins.find(name);
            if (it != plugins.end()) {
                return true;
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
                    return true;
                }
                fatal("Unknown plugin: `{}`\n", name);
                return false;
            }
            return false;
        }
    };

    AKR_EXPORT PluginManager *get_plugin_manager() {
        static PluginManagerImpl manager;
        return &manager;
    }

    std::shared_ptr<Component> create_component(const char *type) {
        auto manager = get_plugin_manager();
        auto plugin = manager->load_plugin(type);
        if (!plugin) {
            error("Failed to create component: `{}`\n", type);
            return nullptr;
        }
        auto ty = Type::get_by_name(type);
        return ty.create_shared().shared_cast<Component>();
    }

} // namespace akari