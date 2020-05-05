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
#include <akari/core/detail/typeinfo.hpp>
#include <akari/core/platform.h>
#include <memory>

namespace akari {
    namespace serialize {
        class InputArchive;
        class OutputArchive;
    } // namespace serialize
    class Component;
    struct Plugin;
    class AKR_EXPORT Component : public std::enable_shared_from_this<Component> {
        bool dirty = true;

      public:
        Component() : dirty(true) {}
        virtual void commit() {}
        virtual void clear_dirty_bit() { dirty = false; }
        virtual void set_dirty_Bit() { dirty = true; }
        virtual bool is_dirty() const { return dirty; }
        virtual TypeInfo type_info() const = 0;
        template <typename Archive> void save(Archive &ar) {}
        template <typename Archive> void load(Archive &ar) {}
    };
    template <typename T, typename U> std::shared_ptr<T> Cast(const std::shared_ptr<U> &p) {
        return std::dynamic_pointer_cast<T>(p);
    }

#define AKR_DECL_COMP()                                                                                     \
    TypeInfo type_info() const override{ return type_of<std::decay_t<decltype(*this)>>(); }                                    \
    friend void _AkariPluginOnLoad(Plugin &);


    using GetPluginFunc = akari::Plugin *(*)();
    AKR_EXPORT std::shared_ptr<Component> create_component(const char *type);

} // namespace akari
#endif // AKARIRENDER_COMPONENT_H
