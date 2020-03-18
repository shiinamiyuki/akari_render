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

#include <Akari/Core/Platform.h>
#include <Akari/Core/Property.hpp>
#include <Akari/Core/Serialize.hpp>
#include <list>
#include <memory>
namespace Akari {
    class Component;
    class IPlugin;
    class AKR_EXPORT Component : public Serializable, std::enable_shared_from_this<Component> {
        bool dirty = true;

      public:
        Component() : dirty(true) {}
        virtual void Accept(PropertyVisitor &visitor) {}
        virtual std::list<Property> GetProperties() { return std::list<Property>(); }
        virtual void Commit() {}
        virtual void ClearDirtyBit() { dirty = false; }
        virtual void SetDirtyBit() { dirty = true; }
        virtual bool IsDirty() const { return dirty; }
    };
#define AKARI_GET_PLUGIN           AkariGetPlugin
#define AKARI_PLUGIN_FUNC_NAME     "AkariGetPlugin"
#define AKR_DECL_COMP(Name, Alias) MYK_TYPE(Name, Alias)

#define AKR_EXPORT_COMP(Name, Interface)                                                                               \
    extern "C" AKR_EXPORT IPlugin *AKARI_GET_PLUGIN() {                                                                \
        struct ThisPlugin : IPlugin {                                                                                  \
            TypeInfo *GetTypeInfo() override { return Name::staticType(); }                                            \
            const char *GetInterface() override {                                                                      \
                return Interface == nullptr || strlen(Interface) == 0 ? "Default" : Interface;                         \
            }                                                                                                          \
        };                                                                                                             \
        static ThisPlugin plugin;                                                                                      \
        return &plugin;                                                                                                \
    }

    using GetPluginFunc = IPlugin *(*)();
    AKR_EXPORT Ptr<Component> CreateComponent(const char * type);


} // namespace Akari
#endif // AKARIRENDER_COMPONENT_H
