
// MIT License
//
// Copyright (c) 2020 椎名深雪
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
#include <optional>
#include <akari/asl/ast.h>
namespace akari::asl {

    class AKR_EXPORT Parser {
        class Impl;
        std::shared_ptr<Impl> impl;

      public:
        Parser();
        ast::TopLevel operator()(const std::string &filename, const std::string &src);
    };
    template <typename K, typename V> struct EnvironmentFrame {
        std::unordered_map<K, V> map;
        std::shared_ptr<EnvironmentFrame<K,V>> parent;
        std::optional<V> at(const K &k) {
            if (map.count(k)) {
                return map.at(k);
            }
            if (parent) {
                return parent->at(k);
            }
            return std::nullopt;
        }
        void insert(const K &k, const V &v) { map.emplace(k, v); }
    };
    template <typename K, typename V> struct Environment {
        std::shared_ptr<EnvironmentFrame<K, V>> frame;
        Environment() { frame = std::make_shared<EnvironmentFrame<K, V>>(); }
        void _push() {
            auto prev = frame;
            frame = std::make_shared<EnvironmentFrame<K, V>>();
            frame->parent = prev;
        }
        void _pop() { frame = frame->parent; }
        struct Guard {
            Environment &self;
            ~Guard() { self._pop(); }
        };
        Guard push() {
            _push();
            return Guard { *this };
        }
        std::optional<V> at(const K &k) { return frame->at(k); }
        void insert(const K &k, const V &v) { frame->insert(k, v); }
    };
} // namespace akari::asl