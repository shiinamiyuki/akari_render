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

#ifndef AKARIRENDER_REFLECT_MACROS_H
#define AKARIRENDER_REFLECT_MACROS_H

#include <iostream>

#include <utility>
#include <vector>

namespace Akari {

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
            if (s[pos] == '"') {
                pos++;
            }

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
    template <typename Visitor> struct ReflectionAccept : Visitor {
        // this is slow
        // but easy to implement
        template <size_t N, typename... Args> void FirstPass(const char (&args_s)[N], Args &&... args) {
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
        template <typename T> void ProcessOne(const char *s, T &&t) {
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
                Visitor::visit(name.c_str(), t, attr);
            } else {
                Visitor::visit(s, t);
            }
        }
        void SecondPass(const char **args_s) {}
        template <typename T, typename... Args> void SecondPass(const char **args_s, T &&t, Args &&... args) {
            ProcessOne(*args_s, t);
            SecondPass(args_s + 1, std::forward<Args>(args)...);
        }
    };
#define AKR_ATTR(var, ...) var // yes it does nothing

    struct GetPropertyVisitor {
        std::vector<Property> props;
        template <typename T> void visit(const char *name, T &&value) { props.emplace_back(name, Attributes(), value); }
        template <typename T> void visit(const char *name, T &&value, const Attributes &attr) {
            props.emplace_back(name, attr, value);
        }
    };

#define AKR_PROPS(...)                                                                                                 \
    std::vector<Property> GetProperties() {                                                                            \
        ReflectionAccept<GetPropertyVisitor> visitor;                                                                  \
        visitor.FirstPass(#__VA_ARGS__, __VA_ARGS__);                                                                  \
        return std::move(visitor.props);                                                                               \
    }                                                                                                                  \
    std::vector<Property> GetProperties() const {                                                                      \
        ReflectionAccept<GetPropertyVisitor> visitor;                                                                  \
        visitor.FirstPass(#__VA_ARGS__, __VA_ARGS__);                                                                  \
        return std::move(visitor.props);                                                                               \
    }
#define AKR_COMP_PROPS(...)                                                                                            \
    std::vector<Property> GetProperties() override {                                                                   \
        ReflectionAccept<GetPropertyVisitor> visitor;                                                                  \
        visitor.FirstPass(#__VA_ARGS__, __VA_ARGS__);                                                                  \
        return std::move(visitor.props);                                                                               \
    }                                                                                                                  \
    std::vector<Property> GetProperties() const override {                                                             \
        ReflectionAccept<GetPropertyVisitor> visitor;                                                                  \
        visitor.FirstPass(#__VA_ARGS__, __VA_ARGS__);                                                                  \
        return std::move(visitor.props);                                                                               \
    }
} // namespace Akari
#endif // AKARIRENDER_REFLECT_MACROS_H
