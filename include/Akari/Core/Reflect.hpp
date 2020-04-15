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

#pragma once

#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Akari {
    struct Type {
        const char *name = nullptr;
        bool operator==(const Type &rhs) const { return name == rhs.name || strcmp(name, rhs.name) == 0; }
    };

    template <typename T> Type Typeof() {
        Type type{typeid(T).name()};
        return type;
    }

    struct Any {
        template <typename T> T &as() {
            auto ty = Typeof<T>;
            if (ty == type) {
                return *reinterpret_cast<T *>(_ptr.get());
            } else {
                throw std::runtime_error("fuck");
            }
        }
        template <typename T> const T &as() const {
            auto ty = Typeof<T>;
            if (ty == type) {
                return *reinterpret_cast<const T *>(_ptr.get());
            } else {
                throw std::runtime_error("fuck");
            }
        }
        Any() = default;
        template <typename T> Any(const T &value) : type(Typeof<T>()) { _ptr = std::make_shared<T>(value); }
        template <typename T> Any &operator=(const T &value) {
            type = Typeof<T>();
            _ptr = std::make_shared<T>(value);
            return *this;
        }
        template <typename T>[[nodiscard]] bool is_of() const { return Typeof<T>() == type; }
        template <typename Visitor, typename... Ts> bool accept(Visitor &&vis) const {
            return _accept<Ts...>(std::forward<Visitor>(vis));
        }

      private:
        template <typename Visitor> bool _accept(Visitor &&vis) const { return false; }
        template <typename Visitor, typename T, typename... Ts> bool _accept(Visitor &&vis) const {
            if (is_of<T>()) {
                vis(as<T>());
                return true;
            }
            return _accept<Ts...>(std::forward<Visitor>(vis));
        }

        Type type;
        std::shared_ptr<void> _ptr;
    };

    struct AnyReference {
        template <typename T> T &as() {
            if (is_const) {
                throw std::runtime_error("Reference is const");
            }
            auto ty = Typeof<T>;
            if (ty == type) {
                return *reinterpret_cast<T *>(_ptr);
            } else {
                throw std::runtime_error("fuck");
            }
        }
        template <typename T> const T &as() const {
            auto ty = Typeof<T>;
            if (ty == type) {
                if (is_const) {
                    return *reinterpret_cast<const T *>(_c_ptr);
                } else {
                    return *reinterpret_cast<const T *>(_ptr);
                }
            } else {
                throw std::runtime_error("fuck");
            }
        }
        AnyReference() = default;
        template <typename T> AnyReference(T &value) : type(Typeof<T>()) { _ptr = &value; }
        template <typename T> AnyReference(const T &value) : type(Typeof<T>()) {
            is_const = true;
            _c_ptr = &value;
            _ptr = nullptr;
        }
        template <typename T> AnyReference &operator=(T &value) {
            type = Typeof<T>();
            _ptr = &value;
            _c_ptr = nullptr;
            return *this;
        }
        template <typename T>[[nodiscard]] bool is_of() const { return Typeof<T>() == type; }
        template <typename Visitor, typename... Ts> bool accept(Visitor &&vis) const {
            return _accept<Ts...>(std::forward<Visitor>(vis));
        }

      private:
        template <typename Visitor> bool _accept(Visitor &&vis) const { return false; }
        template <typename Visitor, typename T, typename... Ts> bool _accept(Visitor &&vis) const {
            if (is_of<T>()) {
                vis(as<T>());
                return true;
            }
            return _accept<Ts...>(std::forward<Visitor>(vis));
        }

        bool is_const = false;
        Type type;
        void *_ptr = nullptr;
        const void *_c_ptr = nullptr;
    };

    using Attributes = std::unordered_map<std::string, std::string>;

    struct Property : AnyReference {

        template <typename T>
        Property(const char *name, Attributes attr, T &value)
            : AnyReference(value), _attr(std::move(attr)), _name(name) {}
        template <typename T>
        Property(const char *name, Attributes attr, const T &value)
            : AnyReference(value), _attr(std::move(attr)), _name(name) {}

        template <typename T> void set(const T &value) {
            auto &ref = as<T>();
            ref = value;
        }
        const char *name() const { return _name.c_str(); }
        const Attributes &attr() const { return _attr; }

      private:
        Attributes _attr;
        std::string _name;
    };

    class Reflect {
      public:
        [[nodiscard]] virtual Type GetType() const = 0;
        [[nodiscard]] virtual std::vector<Property> GetProperties() const { return {}; }
        virtual std::vector<Property> GetProperties() { return {}; }
    };

} // namespace Akari