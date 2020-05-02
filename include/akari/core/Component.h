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

#ifndef AKARIRENDER_COMPONENT_H
#define AKARIRENDER_COMPONENT_H
#include <akari/core/akari.h>
#include <akari/core/platform.h>
#include <akari/core/reflect.hpp>
#include <akari/core/serialize.hpp>
#include <list>
#include <memory>
#include <mutex>
namespace akari {
    class Component;
    class Plugin;

    class AKR_EXPORT Component : public Serializable, public std::enable_shared_from_this<Component> {
        bool dirty = true;

      public:
        Component() : dirty(true) {}
        virtual void commit() {}
        virtual void ClearDirtyBit() { dirty = false; }
        virtual void SetDirtyBit() { dirty = true; }
        virtual bool IsDirty() const { return dirty; }
    };
    template <typename T, typename U> std::shared_ptr<T> Cast(const std::shared_ptr<U> &p) {
        return std::dynamic_pointer_cast<T>(p);
    }
#define AKARI_GET_PLUGIN       AkariGetPlugin
#define AKARI_PLUGIN_FUNC_NAME "AkariGetPlugin"
#define AKR_PLUGIN_ON_LOAD                                                                                             \
    void _AkariPluginOnLoad();                                                                                         \
    extern "C" AKR_EXPORT void AkariPluginOnLoad() { _AkariPluginOnLoad(); }                                           \
    void _AkariPluginOnLoad()
#define AKR_STATIC_CLASS(CLASS, ALIAS)                                                                                 \
    static akari::Class *StaticClass() {                                                                               \
        static std::unique_ptr<Class> _class;                                                                          \
        static std::once_flag flag;                                                                                    \
        std::call_once(flag, [&]() {                                                                                   \
            _class = std::make_unique<Class>(                                                                          \
                ALIAS, []() { return std::make_shared<CLASS>(); },                                                     \
                [](Serializable &self, Serialize::InputArchive &ar) { dynamic_cast<CLASS &>(self).load(ar); },         \
                [](const Serializable &self, Serialize::OutputArchive &ar) {                                           \
                    dynamic_cast<const CLASS &>(self).save(ar);                                                        \
                });                                                                                                    \
        });                                                                                                            \
        return _class.get();                                                                                           \
    }                                                                                                                  \
    akari::Class *GetClass() const override { return StaticClass(); }

#define AKR_DECL_COMP(Name, Alias)                                                                                     \
    AKR_STATIC_CLASS(Name, Alias)                                                                                      \
    friend void _AkariPluginOnLoad();

#define AKR_EXPORT_COMP(Name, Interface)                                                                               \
    extern "C" AKR_EXPORT Plugin *AKARI_GET_PLUGIN() {                                                                 \
        struct ThisPlugin : Plugin {                                                                                   \
            Class *GetClass() const override { return Name::StaticClass(); }                                           \
            const char *GetInterface() const override { return strlen(Interface) == 0 ? "Default" : Interface; }       \
        };                                                                                                             \
        static ThisPlugin plugin;                                                                                      \
        return &plugin;                                                                                                \
    }

    using GetPluginFunc = akari::Plugin *(*)();
    AKR_EXPORT std::shared_ptr<Component> CreateComponent(const char *type);

} // namespace akari
#endif // AKARIRENDER_COMPONENT_H
