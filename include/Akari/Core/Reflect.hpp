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

#include <Akari/Core/Platform.h>
#include <cstring>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Akari {
    struct TypeInfo {
        const char *name = nullptr;
        bool operator==(const TypeInfo &rhs) const { return name == rhs.name || strcmp(name, rhs.name) == 0; }
        bool operator!=(const TypeInfo &rhs) const { return !(rhs == *this); }
    };

    template <typename T> TypeInfo type_of() {
        TypeInfo type{typeid(T).name()};
        return type;
    }
    template <typename T> TypeInfo type_of(T &&v) {
        TypeInfo type{typeid(v).name()};
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
        template <typename T> Any(from_value_t _, T value) : type(type_of<T>()), kind(EValue) {
            _ptr = make_container<T>(std::move(value));
        }
        template <typename T, typename U = std::remove_reference_t<T>>
        Any(from_ref_t _, T &&value) : type(type_of<std::decay_t<T>>()), kind(ERef) {
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
            return type == type_of<T>();
        }
        [[nodiscard]] bool has_value() const { return _ptr != nullptr; }
        template <typename T> T &as() {
            if (kind == EVoid) {
                throw std::runtime_error("Any is of void");
            }
            if (type != type_of<T>()) {
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

      private:
        TypeInfo type;
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
        std::vector<TypeInfo> signature;

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
        [[nodiscard]] const std::vector<TypeInfo> &get_signature() const { return signature; }

      private:
        template <typename R, typename... Args> void _from_lambda(std::function<R(Args...)> &&f) {
            _from<std::function<R(Args...)>, R, Args...>(std::move(f));
        }
        template <typename F, typename R, typename... Args> void _from(F &&f) {
            signature = {type_of<Args>()...};
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
    namespace detail {
        struct meta_instance;
    }
    struct Property {
        friend struct detail::meta_instance;
        [[nodiscard]] const char *name() const { return _name.data(); }
        [[nodiscard]] const Attributes &attr() const { return _attr.get(); }
        Property(const char *name, const Attributes &attr) : _name(name), _attr(attr) {}
        Any get(Any &any) { return _get(any); }
        void set(Any &obj, const Any &value) { _set(obj, value); }

      private:
        std::function<void(Any &, const Any &)> _set;
        std::function<Any(Any &)> _get;
        std::string_view _name;
        std::reference_wrapper<const Attributes> _attr;
    };

    namespace detail {
        struct meta_instance {
            using GetFieldFunc = std::function<Any(const Any &)>;
            std::unordered_map<std::string, Property> properties;
            std::unordered_map<std::string, Attributes> attributes;

            template <typename T, typename U> meta_instance &field(const char *name, T U::*p) {
                auto it = attributes.find(name);
                if (it == attributes.end()) {
                    attributes.insert(std::make_pair(name, Attributes()));
                }
                auto &attr = attributes.at(name);
                auto get = [=](Any &any) -> Any {
                    auto &object = any.as<U>();

                    return make_any_ref(object.*p);
                };
                auto set = [=](Any &any, const Any &value) {
                    auto &object = any.as<U>();
                    object.*p = value.as<T>();
                };
                properties.emplace(std::make_pair(name, Property(name, attr)));
                properties.at(name)._get = get;
                properties.at(name)._set = set;
                return *this;
            }

            meta_instance &add_attribute(const char *name, const char *key, const char *value) {
                attributes[name][key] = value;
                return *this;
            }
        };
        struct AKR_EXPORT reflection_manager {
            std::unordered_map<std::string, meta_instance> instances;
            static reflection_manager &instance();
        };

        template <typename T> struct get_internal { using type = T; };
        template <typename T> struct get_internal<T *> { using type = T; };
        template <typename T> struct get_internal<std::shared_ptr<T>> { using type = T; };
        template <typename T> using get_internal_t = typename get_internal<T>::type;

        template <typename T> struct is_shared_ptr : std::false_type {};
        template <typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
        template <typename T> constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

    } // namespace detail

    template <typename T> detail::meta_instance &register_type() {
        auto type = type_of<T>();
        auto &mgr = detail::reflection_manager::instance();
        mgr.instances[type.name] = detail::meta_instance();
        return mgr.instances[type.name];
    }
    struct Type {
        template <typename T> struct _tag {};
        template <typename T> static const Type &get() {
            static Type _this_type(_tag<T>{});
            return _this_type;
        }
        [[nodiscard]] std::vector<Property> get_properties() const {
            std::vector<Property> v;
            _foreach([&](auto prop) { v.emplace_back(prop); });
            return v;
        }

      private:
        template <typename T> explicit Type(_tag<T>) {
            _get = [](const char *name) {
                auto &mgr = detail::reflection_manager::instance();
                auto &instance = mgr.instances.at(type_of<T>().name);
                return instance.properties.at(name);
            };
            _foreach = [](const std::function<void(Property)> &f) {
                auto &mgr = detail::reflection_manager::instance();
                auto &instance = mgr.instances.at(type_of<T>().name);
                for (auto &field : instance.properties) {
                    f(field.second);
                }
            };
        }
        std::function<Property(const char *name)> _get;
        std::function<void(const std::function<void(Property)> &)> _foreach;
    };

} // namespace Akari