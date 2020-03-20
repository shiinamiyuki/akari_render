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

#ifndef AKARIRENDER_PROPERTY_HPP
#define AKARIRENDER_PROPERTY_HPP

#include <Akari/Core/Akari.h>
#include <Akari/Core/Component.h>
#include <Akari/Core/Math.h>
#include <variant>
#include <memory>
#include <string>
#include <functional>
namespace Akari {
    class Component;
    template <typename T> using Ptr = std::shared_ptr<T>;
    template <typename T, typename U> Ptr<T> Cast(const Ptr<U> &p) { return std::dynamic_pointer_cast<T>(p); }
    template <typename... T> using RefVariant = std::variant<std::reference_wrapper<T>...>;
    //    using PropertyValue =
    //        RefVariant<bool, int, float, vec2, vec3, ivec2, ivec3, std::string, std::shared_ptr<Component>>;
    template <typename... T> struct TPropertyTypes {
        using RefValue = RefVariant<T...>;
        using Value = std::variant<T...>;
    };
    using PropertyTypes =
        TPropertyTypes<bool, int, float, vec2, vec3, ivec2, ivec3, std::string, std::shared_ptr<Component>>;
    struct Property {
        std::weak_ptr<Component> object;

        template <typename T> Property(const char *name, T &v) : name(name) { Load(v); }
        [[nodiscard]] const char *GetName() const { return name; }
        [[nodiscard]] const PropertyTypes::Value &GetValue() const { return value; }
        template <typename T> void SetValue(T v) { setValueCallback(PropertyTypes::Value(std::move(v))); }

      private:
        const char *name = nullptr;
        PropertyTypes::Value value;
        template <typename T, typename _ = void> struct is_of_component : std::false_type {};
        template <typename T>
        struct is_of_component<T, std::enable_if_t<std::is_base_of_v<Component, T>, std::void_t<T>>> : std::true_type {
        };
        template <typename T> inline static constexpr bool is_of_component_v = is_of_component<T>::value;

        template <typename T> std::enable_if_t<!is_of_component_v<T>, void> Load(T &v) {
            value = v;
            setValueCallback = [=](const PropertyTypes::Value &newVal) {
                std::visit(
                    [&](auto &&item) {
                        using U = std::decay_t<decltype(item)>;
                        if constexpr (std::is_convertible_v<U, T>) {
                            v = (T)item;
                            value = v;
                        } else if constexpr (std::is_constructible_v<T, U>) {
                            v = T(item);
                            value = v;
                        } else {
                            AKARI_PANIC("Invalid Type for Property");
                        }
                    },
                    newVal);
            };
        }
        template <typename T> std::enable_if_t<is_of_component_v<T>, void> Load(std::shared_ptr<T> &v) {
            value = std::shared_ptr<Component>(v);
            setValueCallback = [=](const PropertyTypes::Value &newVal) {
                auto ptr = Cast<T>(std::get<std::shared_ptr<Component>>(newVal));
                value = ptr;
                v = ptr;
            };
        }

        std::function<void(const PropertyTypes::Value &)> setValueCallback;
    };

    class PropertyVisitor {
      public:
        virtual void visit(Property &property) = 0;
    };

} // namespace Akari
#endif // AKARIRENDER_PROPERTY_HPP
