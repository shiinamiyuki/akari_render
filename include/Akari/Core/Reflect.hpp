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
#include <iostream>
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

      private:
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
        }
        template <typename T> AnyReference &operator=(T &value) {
            type = Typeof<T>();
            _ptr = &value;
            return *this;
        }

      private:
        bool is_const = false;
        Type type;
        void *_ptr = nullptr;
        const void *_c_ptr = nullptr;
    };

    struct Class {
        [[nodiscard]] virtual Any Create() const = 0;
        [[nodiscard]] virtual Type GetType() const = 0;
        [[nodiscard]] const char *name() const { return GetType().name; }
        bool operator==(const Class &rhs) const { return GetType() == rhs.GetType(); }
    };

    using Attributes = std::unordered_map<std::string, std::string>;

    inline Attributes ParseAttributes(const char *s) {
        Attributes attr;
        size_t pos = 0;
        auto skip_space = [&]() {
            while (pos < strlen(s) && isspace(s[pos])) {
                pos++;
            }
        };
        auto identifier = [&]() {
            std::string iden;
            while (pos < strlen(s) && isalnum(s[pos])) {
                iden += s[pos];
                pos++;
            }
            return iden;
        };
        auto string = [&]() {
            std::string str;
            if (s[pos] == '"')
                pos++;
            while (pos < strlen(s) && ('"' != s[pos])) {
                str += s[pos];
                pos++;
            }
            pos++;
            return str;
        };
        while (pos < strlen(s)) {
            skip_space();
            if (s[pos] == ')')
                break;
            auto key = identifier();
            skip_space();
            if (s[pos] == '=')
                pos++;
            else {
                std::cout << (s + pos) << std::endl;
                throw std::runtime_error("wtf");
            }
            skip_space();
            auto value = string();
            attr[key] = value;
        }
        return attr;
    }
    struct Property : AnyReference {

        template <typename T>
        Property(const char *name, const Attributes &attr, T &value) : AnyReference(value), _attr(attr), _name(name) {}
        template <typename T>
        Property(const char *name, const Attributes &attr, const T &value)
            : AnyReference(value), _attr(attr), _name(name) {}

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

    template<typename Visitor>
    struct ReflectionAccept {
        // this is slow
        // but easy to implement
        template <size_t N, typename... Args> static void FirstPass(const char (&args_s)[N], Args &&... args) {
            std::string s = args_s;
            std::array<std::string, sizeof...(Args)> array;
            size_t pos = 0;

            for (size_t i = 0; i < array.size(); i++) {
                auto skip_space = [&]() {
                    while (pos < s.length() && isspace(s[pos])) {
                        pos++;
                    }
                };
                skip_space();
                while (pos < s.length() && s[pos] != ',') {
                    if (s[pos] == '(') {
                        while (pos < s.length() && s[pos] != ')') {
                            array[i] += s[pos++];
                        }
                    } else
                        array[i] += s[pos++];
                }
                pos++;
                skip_space();
            }
            const char *a[sizeof...(Args)];
            for (size_t i = 0; i < array.size(); i++) {
                a[i] = array[i].c_str();
            }
            SecondPass<Args...>(a, std::forward<Args>(args)...);
        }
        template <typename T> static void ProcessOne(const char *s, T &&t) {
            if (strncmp(s, "AKR_ATTR", strlen("AKR_ATTR")) == 0) {
                s = s + strlen("AKR_ATTR");
                std::string inside = s;
                Attributes attr;
                size_t pos = 0;
                auto skip_space = [&]() {
                    while (pos < inside.length() && isspace(inside[pos])) {
                        pos++;
                    }
                };
                skip_space();
                if (inside[pos] == '(') {
                    pos++;
                } else {
                    throw std::runtime_error("wtf");
                }
                skip_space();
                std::string name;
                while (pos < inside.length() && isalnum(inside[pos])) {
                    name += inside[pos++];
                }
                // std::cout << "name "<< name << std::endl;
                skip_space();
                if (pos < inside.length()) {
                    if (inside[pos] == ',') {
                        pos++;
                    }
                    skip_space();

                    if (pos < inside.length()) {
                        attr = ParseAttributes(s + pos);
                    }
                }
                Visitor::visit(name.c_str(),  t, attr);
            } else {
                Visitor::visit(s, t);
            }
        }
        void static SecondPass(const char **args_s) {}
        template <typename T, typename... Args> static void SecondPass(const char **args_s, T &&t, Args &&... args) {
            ProcessOne(*args_s, t);
            SecondPass(args_s + 1, std::forward<Args>(args)...);
        }
    };
#define AKR_ATTR(var, ...) var // yes it does nothing
#define AKR_PROPS(...)                                                                                                 \
    std::vector<Property> GetProperties() {                                                                            \
        ReflectionVisitor visitor;                                                                                     \
        visitor.FirstPass(#__VA_ARGS__, __VA_ARGS__);                                                                  \
        return std::move(visitor.props);                                                                               \
    }                                                                                                                  \
    std::vector<Property> GetProperties() const {                                                                      \
        ReflectionVisitor visitor;                                                                                     \
        visitor.FirstPass(#__VA_ARGS__, __VA_ARGS__);                                                                  \
        return std::move(visitor.props);                                                                               \
    }
#define AKR_COMP_PROPS(...)                                                                                                 \
    std::vector<Property> GetProperties() override{                                                                            \
        ReflectionVisitor visitor;                                                                                     \
        visitor.FirstPass(#__VA_ARGS__, __VA_ARGS__);                                                                  \
        return std::move(visitor.props);                                                                               \
    }                                                                                                                  \
    std::vector<Property> GetProperties() const override {                                                                      \
        ReflectionVisitor visitor;                                                                                     \
        visitor.FirstPass(#__VA_ARGS__, __VA_ARGS__);                                                                  \
        return std::move(visitor.props);                                                                               \
    }
} // namespace Akari