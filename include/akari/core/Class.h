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

#ifndef AKARIRENDER_CLASS_H
#define AKARIRENDER_CLASS_H

#include <akari/core/platform.h>
#include <akari/core/reflect.hpp>
#include <functional>
#include <string>
namespace akari {
    template <typename T> using Ref = std::shared_ptr<T>;
    template <typename U, typename T> Ref<U> cast(const Ref<T> &p) { return std::dynamic_pointer_cast<U>(p); }
    template <typename U, typename T> std::optional<Ref<U>> safe_cast(const Ref<T> &p) {
        if (!p)
            return nullptr;
        auto q = std::dynamic_pointer_cast<U>(p);
        if (!q) {
            return {};
        }
        return q;
    }
    namespace serialize {
        class OutputArchive;
        class InputArchive;
    } // namespace serialize
    class Object;
    class Serializable;
    class AKR_EXPORT Class {
      public:
        using LoadFunction = std::function<void(Serializable &, serialize::InputArchive &ar)>;
        using SaveFunction = std::function<void(const Serializable &, serialize::OutputArchive &ar)>;
        using DefaultConstructor = std::function<Ref<Object>()>;
        void Load(Serializable &, serialize::InputArchive &ar) const;
        void Save(const Serializable &, serialize::OutputArchive &ar) const;
        [[nodiscard]] TypeInfo GetType() const { return _type; }
        [[nodiscard]] const std::string &GetName() const { return _name; }
        Class(const std::string &name, DefaultConstructor constructor = {}, LoadFunction loader = {},
              SaveFunction saver = {});

        [[nodiscard]] const DefaultConstructor &GetDefaultConstructor() const { return _ctor; }
        [[nodiscard]] Ref<Object> Create() const { return GetDefaultConstructor()(); }

      private:
        std::string _name;
        TypeInfo _type;
        DefaultConstructor _ctor;
        LoadFunction _loader;
        SaveFunction _saver;
    };
    AKR_EXPORT Class *GetClass(const std::string &name);
} // namespace akari
#endif // AKARIRENDER_CLASS_H
