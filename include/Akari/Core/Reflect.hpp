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
    namespace detail {
        template <typename T> struct get_internal { using type = T; };
        template <typename T> struct get_internal<T *> { using type = T; };
        template <typename T> struct get_internal<std::shared_ptr<T>> { using type = T; };
        template <typename T> using get_internal_t = typename get_internal<T>::type;

        template <typename T> struct is_shared_ptr : std::false_type {};
        template <typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
        template <typename T> constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;
        //        template <typename T> struct is_reference_wrapper : std::false_type {};
        //        template <typename T> struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {};
        //        template <typename T> constexpr bool is_reference_wrapper_v = is_reference_wrapper<T>::value;
    } // namespace detail

    struct Type;
    struct Any;
    template <typename T> Any make_any(T value);
    template <typename T> Any make_any_ref(T &&value);
    struct Any {
        friend struct Type;

      private:
        struct Container {
            [[nodiscard]] virtual std::unique_ptr<Container> clone() const = 0;
            virtual void *get() = 0;
            virtual bool is_pointer() const = 0;
            virtual Any get_underlying() = 0;
            virtual ~Container() = default;
        };
        template <typename Actual, typename T> struct ContainerImpl : Container {
            T value;
            explicit ContainerImpl(const T &_value) : value(_value) {}
            [[nodiscard]] std::unique_ptr<Container> clone() const override {
                return std::make_unique<ContainerImpl>(value);
            }
            void *get() override { return &value; }
            [[nodiscard]] bool is_pointer() const override {
                return detail::is_shared_ptr_v<Actual> || std::is_pointer_v<Actual>;
            }
            [[nodiscard]] Any get_underlying() override {
                if constexpr (detail::is_shared_ptr_v<Actual> || std::is_pointer_v<Actual>) {
                    Actual &tmp = (value);
                    return make_any_ref(*tmp);
                }
                return Any();
            }
        };
        template <typename A, typename T> std::unique_ptr<Container> make_container(T &&value) {
            return std::make_unique<ContainerImpl<A, T>>(value);
        }

      public:
        struct from_value_t {};
        struct from_ref_t {};
        Any() = default;
        template <typename T>
        Any(T value, std::enable_if_t<std::is_reference_v<T>, std::true_type> _ = {})
            : Any(from_ref_t{}, std::forward<T>(value)) {}
        template <typename T>
        Any(T value, std::enable_if_t<!std::is_reference_v<T>, std::true_type> _ = {})
            : Any(from_value_t{}, std::forward<T>(value)) {}
        template <typename T> Any(from_value_t _, T value) : type(type_of<T>()), kind(EValue) {
            _ptr = make_container<std::decay_t<T>, T>(std::move(value));
        }
        template <typename T, typename U = std::remove_reference_t<T>>
        Any(from_ref_t _, T &&value) : type(type_of<std::decay_t<T>>()), kind(ERef) {
            using R = std::reference_wrapper<U>;
            _ptr = make_container<std::decay_t<T>, R>(R(std::forward<T>(value)));
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
        [[nodiscard]] bool is_pointer() const { return _ptr->is_pointer(); }
        [[nodiscard]] Any get_underlying() const { return _ptr->get_underlying(); }

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
        template <typename T> struct meta_instance_handle;
    } // namespace detail
    struct Property {
        friend struct detail::meta_instance;
        template <typename T> friend struct detail::meta_instance_handle;
        [[nodiscard]] const char *name() const { return _name.data(); }
        [[nodiscard]] const Attributes &attr() const { return _attr.get(); }
        Property(const char *name, const Attributes &attr) : _name(name), _attr(attr) {}
        Any get(Any &any) {
            if (any.is_pointer()) {
                Any tmp = any.get_underlying();
                return _get(tmp);
            } else {
                return _get(any);
            }
        }
        void set(Any &obj, const Any &value) {
            if (obj.is_pointer()) {
                Any tmp = obj.get_underlying();
                _set(tmp, value);
            } else {
                _set(obj, value);
            }
        }

      private:
        std::function<void(Any &, const Any &)> _set;
        std::function<Any(Any &)> _get;
        std::function<void(Any &, const Any &)> _set_ptr;
        std::function<Any(Any &)> _get_ptr;
        std::string_view _name;
        std::reference_wrapper<const Attributes> _attr;
    };
    namespace detail {

        struct meta_instance {
            using GetFieldFunc = std::function<Any(const Any &)>;
            std::unordered_map<std::string, Property> properties;
            std::unordered_map<std::string, Attributes> attributes;
            std::vector<Function> constructors;
            std::vector<Function> shared_constructors;
        };
        template <typename U> struct meta_instance_handle {
            meta_instance_handle(meta_instance &i)
                : properties(i.properties), attributes(i.attributes), constructors(i.constructors),
                  shared_constructors(i.shared_constructors) {}
            std::unordered_map<std::string, Property> &properties;
            std::unordered_map<std::string, Attributes> &attributes;
            std::vector<Function> &constructors;
            std::vector<Function> &shared_constructors;
            template <typename... Args> meta_instance_handle &constructor() {
                std::function<U(Args...)> ctor = [](Args... args) { return U(std::forward<Args>(args)...); };
                std::function<std::shared_ptr<U>(Args...)> ctor_shared = [](Args... args) {
                    return std::make_shared<U>(std::forward<Args>(args)...);
                };
                constructors.emplace_back(ctor);
                shared_constructors.emplace_back(ctor_shared);
                return *this;
            }

            template <typename T> meta_instance_handle &property(const char *name, T U::*p) {
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

            meta_instance_handle &add_attribute(const char *name, const char *key, const char *value) {
                attributes[name][key] = value;
                return *this;
            }
        };
        struct AKR_EXPORT reflection_manager {
            std::unordered_map<std::string_view, meta_instance> instances;
            std::unordered_map<std::string_view, const char *> name_map;
            static reflection_manager &instance();
        };

    } // namespace detail

    template <typename T> detail::meta_instance_handle<T> register_type(const char *name = nullptr) {
        auto type = type_of<T>();
        auto &mgr = detail::reflection_manager::instance();
        mgr.instances[type.name] = detail::meta_instance();
        if (name) {
            mgr.name_map.emplace(name, type_of<T>().name);
        }
        return detail::meta_instance_handle<T>(mgr.instances[type.name]);
    }
    struct Type {
        template <typename T> struct _tag {};
        template <typename T> static const Type &get() {
            static Type _this_type(_tag<T>{});
            return _this_type;
        }
        static const Type &get_by_name(const char *name) {
            static Type _this_type(detail::reflection_manager::instance().name_map.at(name));
            return _this_type;
        }
        static Type get(const Any &any) { return Type(get_by_typeid(any.type.name)); }
        template <typename T> static Type get_by_typeid(T &&v) {
            Type _this_type(type_of(v).name);
            return _this_type;
        }
        [[nodiscard]] Property get_property(const char *name) const { return _get().properties.at(name); }
        [[nodiscard]] std::vector<Property> get_properties() const {
            std::vector<Property> v;
            for (auto &field : _get().properties) {
                v.emplace_back(field.second);
            }
            return v;
        }
        template <typename... Args>[[nodiscard]] Any create(Args &&... args) const {
            for (auto &ctor : _get().constructors) {
                try {
                    return ctor.invoke(Any(std::forward<Args>(args))...);
                } catch (std::runtime_error &) {
                }
            }
            throw std::runtime_error("no matching constructor");
        }
        template <typename... Args>[[nodiscard]] Any create_shared(Args &&... args) const {
            for (auto &ctor : _get().shared_constructors) {
                try {
                    return ctor.invoke(Any(std::forward<Args>(args))...);
                } catch (std::runtime_error &) {
                }
            }
            throw std::runtime_error("no matching constructor");
        }

      private:
        std::function<detail::meta_instance &(void)> _get;
        explicit Type(const char *type) {
            _get = [=]() -> detail::meta_instance & {
                auto &mgr = detail::reflection_manager::instance();
                auto &instance = mgr.instances.at(type);
                return instance;
            };
        }
        template <typename T> explicit Type(_tag<T>) : Type(type_of<T>().name) {}
    };

} // namespace Akari