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

#include <akari/Core/Logger.h>
#include <akari/Core/Plugin.h>
#ifdef _WIN32

#include <Windows.h>

namespace akari {
    void SharedLibraryLoader::Load(const char *path) { handle = LoadLibraryA(path); }

    SharedLibraryFunc SharedLibraryLoader::GetFuncPointer(const char *name) {
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
    void SharedLibraryLoader::Load(const char *path) { handle = dlopen(path, RTLD_LAZY); }

    SharedLibraryFunc SharedLibraryLoader::GetFuncPointer(const char *name) {
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
        void SetPluginPath(const char *path) override { prefix = path; }
        bool LoadPath(const char *path) override {
            Info("Loading {}\n", path);
            auto lib = std::make_unique<SharedLibraryLoader>(path);
            if (!lib) {
                Error("Cannot load {}\n", path);
                return false;
            }
            if (auto f = lib->GetFuncPointer("AkariPluginOnLoad")) {
                f();
            }
            auto p = lib->GetFuncPointer(AKARI_PLUGIN_FUNC_NAME);
            if (!p) {
                Error("Cannot resolve symbol {} in {}\n", AKARI_PLUGIN_FUNC_NAME, path);
                return false;
            }
            auto plugin = ((GetPluginFunc)p)();
            plugins[plugin->GetClass()->GetName()] = plugin;
            sharedLibraries.emplace_back(std::move(lib));
            return true;
        }

        Plugin *LoadPlugin(const char *name) override {
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
            if (LoadPath(path.string().c_str())) {
                it = plugins.find(name);
                if (it != plugins.end()) {
                    return it->second;
                }
                Fatal("Unknown plugin: `{}`\n", name);
                return nullptr;
            }
            return nullptr;
        }

        void ForeachPlugin(const std::function<void(Plugin *)> &func) override {
            for (auto &el : plugins) {
                func(el.second);
            }
        }
    };

    AKR_EXPORT PluginManager *GetPluginManager() {
        static PluginManagerImpl manager;
        return &manager;
    }

    std::shared_ptr<Component> CreateComponent(const char *type) {
        auto manager = GetPluginManager();
        auto plugin = manager->LoadPlugin(type);
        if (!plugin) {
            Error("Failed to create component: `{}`\n", type);
            return nullptr;
        }
        auto obj = cast<Component>(plugin->GetClass()->Create());
        return obj;
    }

} // namespace akari