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
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Akari {
    struct Type {
        const char *name = nullptr;
        bool operator==(const Type &rhs) const { return name == rhs.name || strcmp(name, rhs.name) == 0; }
        bool operator!=(const Type &rhs) const { return !(rhs == *this); }
    };

    template <typename T> Type Typeof() {
        Type type{typeid(T).name()};
        return type;
    }
    struct Any {
      private:
        struct Container {
            [[nodiscard]] virtual std::unique_ptr<Container> clone() const = 0;
            virtual void *get() = 0;

            virtual ~Container() = default;
        };
        template <typename T> struct ContainerImpl : Container {
            T value;
            explicit ContainerImpl(const T &_value) : value(_value) {}
            [[nodiscard]] std::unique_ptr<Container> clone() const override {
                return std::make_unique<ContainerImpl>(value);
            }
            void *get() override { return &value; }
        };
        template <typename T> std::unique_ptr<Container> make_container(T &&value) {
            return std::make_unique<ContainerImpl<T>>(value);
        }

      public:
        struct from_value_t {};
        struct from_ref_t {};
        Any() = default;
        template <typename T> Any(from_value_t _, T value) : type(Typeof<T>()), kind(EValue) {
            _ptr = make_container<T>(std::move(value));
        }
        template <typename T, typename U = std::remove_reference_t<T>>
        Any(from_ref_t _, T &&value) : type(Typeof<std::decay_t<T>>()), kind(ERef) {
            using R = std::reference_wrapper<U>;
            _ptr = make_container<R>(R(std::forward<T>(value)));
        }
        Any(const Any &rhs) : type(rhs.type), kind(rhs.kind) {
            if (rhs._ptr) {
                _ptr = rhs._ptr->clone();
            }
        }
        Any(Any &&rhs) noexcept : type(rhs.type), kind(rhs.kind) { _ptr = std::move(rhs._ptr); }
        Any &operator=(Any &&rhs) {
            type = rhs.type;
            kind = rhs.kind;
            _ptr = std::move(rhs._ptr);
            return *this;
        }
        Any &operator=(const Any &rhs) {
            if (&rhs == this)
                return *this;
            type = rhs.type;
            kind = rhs.kind;
            if (rhs._ptr) {
                _ptr = rhs._ptr->clone();
            }
            return *this;
        }
        template <typename T>[[nodiscard]] bool is_of() const {
            if (kind == EVoid) {
                return std::is_same_v<T, void>;
            }
            return type == Typeof<T>();
        }
        [[nodiscard]] bool has_value() const { return _ptr != nullptr; }
        template <typename T> T &as() {
            if (kind == EVoid) {
                throw std::runtime_error("Any is of void");
            }
            if (type != Typeof<T>()) {
                throw std::runtime_error("bad Any::as<T>()");
            }

            if (kind == ERef) {
                using result_t = std::reference_wrapper<T>;
                auto raw = _ptr->get();
                auto ref_wrapper = *reinterpret_cast<result_t *>(raw);
                return ref_wrapper.get();
            } else {
                auto raw = _ptr->get();
                // std::cout <<"raw " << raw << std::endl;
                return *reinterpret_cast<T *>(raw);
            }
        }
        template <typename T> const T &as() const { return const_cast<Any *>(this)->as<T>(); }
        // private:
        Type type;
        std::unique_ptr<Container> _ptr;
        enum Kind : uint8_t { EVoid, EValue, ERef };
        Kind kind = EVoid;
    };

    template <typename T> Any make_any(T value) { return Any(Any::from_value_t{}, std::move(value)); }
    template <typename T> Any make_any_ref(T &&value) { return Any(Any::from_ref_t{}, std::forward<T>(value)); }

    template <size_t Idx, typename Tuple> struct get_nth_element {
        using type = typename std::tuple_element<Idx, Tuple>::type;
    };

    template <size_t Idx, typename Tuple> using nth_element_t = typename get_nth_element<Idx, Tuple>::type;

    struct Function {
      private:
        template <typename R, typename T> static Any _make_any_(T &&value) {
            if constexpr (std::is_reference_v<R>) {
                return make_any_ref(value);
            } else {
                return make_any(value);
            }
        }

      public:
        using FunctionWrapper = std::function<Any(std::vector<Any>)>;
        template <typename... Args> Any invoke(Args &&... args) {
            static_assert(std::conjunction_v<std::is_same<Args, Any>...>);
            std::vector<Any> v{args...};
            return wrapper(std::move(v));
        }
        template <typename F, typename = std::enable_if_t<std::is_class_v<F>>> explicit Function(F &&f) {
            std::function _f = f;
            _from_lambda(std::move(_f));
        }
        template <typename T, typename R, typename... Args> explicit Function(R (T::*f)(Args...)) {
            std::function _method = [=](T *obj, Args... args) -> R { return (obj->*f)(args...); };
            _from<decltype(_method), R, T *, Args...>(std::move(_method));
        }
        template <typename R, typename... Args> explicit Function(const std::function<R(Args...)> &f) {
            _from<decltype(f), R, Args...>(f);
        }
        template <typename R, typename... Args> explicit Function(R (*f)(Args...)) {
            _from<decltype(f), R, Args...>(f);
        }

      private:
        template <typename R, typename... Args> void _from_lambda(std::function<R(Args...)> &&f) {
            _from<std::function<R(Args...)>, R, Args...>(std::move(f));
        }
        template <typename F, typename R, typename... Args> void _from(F &&f) {
            wrapper = [=](std::vector<Any> arg) -> Any {
                using arg_list_t = std::tuple<Args...>;
                // clang-format off
              constexpr auto nArgs = sizeof...(Args);
              if(arg.size() != nArgs){
                  throw std::runtime_error("argument count mismatch");
              }
              static_assert(nArgs <= 8, "at most 8 args are supported");
              if constexpr (!std::is_same_v<R,void>){
                  if constexpr (nArgs == 0){
                      return _make_any_<R>(f());
                  }else if constexpr (nArgs == 1){
                      return _make_any_<R>(f(arg[0].as<nth_element_t<0, arg_list_t>>()));
                  }else if constexpr (nArgs == 2){
                      return _make_any_<R>(f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                                             arg[1].as<nth_element_t<1, arg_list_t>>()));
                  }else if constexpr (nArgs == 3){
                      return _make_any_<R>(f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                                             arg[1].as<nth_element_t<1, arg_list_t>>(),
                                             arg[2].as<nth_element_t<2, arg_list_t>>()));
                  }else if constexpr (nArgs == 4){
                      return _make_any_<R>(f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                                             arg[1].as<nth_element_t<1, arg_list_t>>(),
                                             arg[2].as<nth_element_t<2, arg_list_t>>(),
                                             arg[3].as<nth_element_t<3, arg_list_t>>()));
                  }else if constexpr (nArgs == 5){
                      return _make_any_<R>(f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                                             arg[1].as<nth_element_t<1, arg_list_t>>(),
                                             arg[2].as<nth_element_t<2, arg_list_t>>(),
                                             arg[3].as<nth_element_t<3, arg_list_t>>(),
                                             arg[4].as<nth_element_t<4, arg_list_t>>()));
                  }else if constexpr (nArgs == 6){
                      return _make_any_<R>(f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                                             arg[1].as<nth_element_t<1, arg_list_t>>(),
                                             arg[2].as<nth_element_t<2, arg_list_t>>(),
                                             arg[3].as<nth_element_t<3, arg_list_t>>(),
                                             arg[4].as<nth_element_t<4, arg_list_t>>(),
                                             arg[5].as<nth_element_t<5, arg_list_t>>()));
                  }else if constexpr (nArgs == 7){
                      return _make_any_<R>(f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                                             arg[1].as<nth_element_t<1, arg_list_t>>(),
                                             arg[2].as<nth_element_t<2, arg_list_t>>(),
                                             arg[3].as<nth_element_t<3, arg_list_t>>(),
                                             arg[4].as<nth_element_t<4, arg_list_t>>(),
                                             arg[5].as<nth_element_t<5, arg_list_t>>(),
                                             arg[6].as<nth_element_t<6, arg_list_t>>()));
                  }else{
                      return _make_any_<R>(f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                                             arg[1].as<nth_element_t<1, arg_list_t>>(),
                                             arg[2].as<nth_element_t<2, arg_list_t>>(),
                                             arg[3].as<nth_element_t<3, arg_list_t>>(),
                                             arg[4].as<nth_element_t<4, arg_list_t>>(),
                                             arg[5].as<nth_element_t<5, arg_list_t>>(),
                                             arg[6].as<nth_element_t<6, arg_list_t>>(),
                                             arg[7].as<nth_element_t<7, arg_list_t>>()));
                  }
              }else{
                  if constexpr (nArgs == 0){
                      (f());return Any();
                  }else if constexpr (nArgs == 1){
                      (f(arg[0].as<nth_element_t<0, arg_list_t>>()));return Any();
                  }else if constexpr (nArgs == 2){
                      (f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                         arg[1].as<nth_element_t<1, arg_list_t>>()));return Any();
                  }else if constexpr (nArgs == 3){
                      (f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                         arg[1].as<nth_element_t<1, arg_list_t>>(),
                         arg[2].as<nth_element_t<2, arg_list_t>>()));return Any();
                  }else if constexpr (nArgs == 4){
                      (f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                         arg[1].as<nth_element_t<1, arg_list_t>>(),
                         arg[2].as<nth_element_t<2, arg_list_t>>(),
                         arg[3].as<nth_element_t<3, arg_list_t>>()));return Any();
                  }else if constexpr (nArgs == 5){
                      (f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                         arg[1].as<nth_element_t<1, arg_list_t>>(),
                         arg[2].as<nth_element_t<2, arg_list_t>>(),
                         arg[3].as<nth_element_t<3, arg_list_t>>(),
                         arg[4].as<nth_element_t<4, arg_list_t>>()));return Any();
                  }else if constexpr (nArgs == 6){
                      (f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                         arg[1].as<nth_element_t<1, arg_list_t>>(),
                         arg[2].as<nth_element_t<2, arg_list_t>>(),
                         arg[3].as<nth_element_t<3, arg_list_t>>(),
                         arg[4].as<nth_element_t<4, arg_list_t>>(),
                         arg[5].as<nth_element_t<5, arg_list_t>>()));return Any();
                  }else if constexpr (nArgs == 7){
                      (f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                         arg[1].as<nth_element_t<1, arg_list_t>>(),
                         arg[2].as<nth_element_t<2, arg_list_t>>(),
                         arg[3].as<nth_element_t<3, arg_list_t>>(),
                         arg[4].as<nth_element_t<4, arg_list_t>>(),
                         arg[5].as<nth_element_t<5, arg_list_t>>(),
                         arg[6].as<nth_element_t<6, arg_list_t>>()));return Any();
                  }else{
                      (f(arg[0].as<nth_element_t<0, arg_list_t>>(),
                         arg[1].as<nth_element_t<1, arg_list_t>>(),
                         arg[2].as<nth_element_t<2, arg_list_t>>(),
                         arg[3].as<nth_element_t<3, arg_list_t>>(),
                         arg[4].as<nth_element_t<4, arg_list_t>>(),
                         arg[5].as<nth_element_t<5, arg_list_t>>(),
                         arg[6].as<nth_element_t<6, arg_list_t>>(),
                         arg[7].as<nth_element_t<7, arg_list_t>>()));return Any();
                  }
              }
                // clang-format on
            };
        }

        FunctionWrapper wrapper;
    };
    using Attributes = std::unordered_map<std::string, std::string>;

    struct Property : Any {
        template <typename T>
        Property(const char *name, Attributes attr, T &value)
            : Any(Any::from_ref_t{}, value), _attr(std::move(attr)), _name(name) {}
        template <typename T>
        Property(const char *name, Attributes attr, const T &value)
            : Any(Any::from_ref_t{}, value), _attr(std::move(attr)), _name(name) {}

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