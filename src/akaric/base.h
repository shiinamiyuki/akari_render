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
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <optional>
#include <akari/core/akari.h>
#include <akaric/panic.h>
namespace akari::asl {

    template <typename T>
    struct is_shared_ptr : std::false_type {};
    template <typename T>
    struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
    class Base : public std::enable_shared_from_this<Base> {
      public:
        template <typename T>
        bool isa() const {
            static_assert(is_shared_ptr<T>::value);
            using U = typename T::element_type;

            return dynamic_cast<const U *>(this) != nullptr;
        }
        template <typename T>
        T cast() const {
            static_assert(is_shared_ptr<T>::value);
            using U = typename T::element_type;
            return std::dynamic_pointer_cast<const U>(shared_from_this());
        }
        template <typename T>
        T cast() {
            static_assert(is_shared_ptr<T>::value);
            using U = typename T::element_type;
            return std::dynamic_pointer_cast<U>(shared_from_this());
        }
        virtual bool is_parent_of(const std::shared_ptr<Base> &ptr) const = 0;
        virtual std::string type_name() const = 0;
        virtual ~Base() = default;
    };
#define AKR_DECL_NODE(Type)                                                                                            \
    std::string type_name() const { return #Type; }                                                                    \
    bool is_parent_of(const std::shared_ptr<Base> &ptr) const { return ptr->isa<std::shared_ptr<Type>>(); }
    template <typename K, typename V, class Hash, class KeyEqual>
    struct EnvironmentFrame {
        std::unordered_map<K, V, Hash, KeyEqual> map;
        std::shared_ptr<EnvironmentFrame<K, V, Hash, KeyEqual>> parent;
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

    template <typename K, typename V, class Hash = std::hash<K>, class KeyEqual = std::equal_to<K>>
    struct Environment {
        std::shared_ptr<EnvironmentFrame<K, V, Hash, KeyEqual>> frame;
        Environment() { frame = std::make_shared<EnvironmentFrame<K, V, Hash, KeyEqual>>(); }
        void _push() {
            auto prev = frame;
            frame = std::make_shared<EnvironmentFrame<K, V, Hash, KeyEqual>>();
            frame->parent = prev;
        }
        void _pop() { frame = frame->parent; }
        struct Guard {
            Environment &self;
            ~Guard() { self._pop(); }
        };
        Guard push() {
            _push();
            return Guard{*this};
        }
        std::optional<V> at(const K &k) { return frame->at(k); }
        void insert(const K &k, const V &v) { frame->insert(k, v); }
    };
} // namespace akari::asl