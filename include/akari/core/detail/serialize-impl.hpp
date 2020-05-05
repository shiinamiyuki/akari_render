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

#ifndef AKARIRENDER_SERIALIZE_IMPL_HPP
#define AKARIRENDER_SERIALIZE_IMPL_HPP

#include <akari/core/reflect.hpp>
#include <json.hpp>
#include <magic_enum.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace akari::serialize {
    using namespace nlohmann;
    class NoSuchKeyError : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    class NoSuchTypeError : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    class DowncastError : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    class SerializationError : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };
    class InputArchive;
    class OutputArchive;
    namespace detail {

        template <class T, typename = void> struct has_member_save : std::false_type {};

        template <class T>
        struct has_member_save<T,
                               std::void_t<decltype(std::declval<const T &>().save(std::declval<OutputArchive &>()))>>
            : std::true_type {};
        template <class T, typename = void> struct has_member_load : std::false_type {};

        template <class T>
        struct has_member_load<T, std::void_t<decltype(std::declval<T &>().load(std::declval<InputArchive &>()))>>
            : std::true_type {};

        template <class T> struct is_json_seriailzable_to_json {
            struct No {};

            template <class U>
            static auto test(int) -> decltype(to_json(std::declval<json &>(), std::declval<const U &>()));

            template <class U> static auto test(...) -> No;

            static const bool value = !std::is_same_v<No, decltype(test<T>(0))>;
        };

        template <class T> struct is_json_seriailzable_adl_to_json {
            struct No {};

            template <class U>
            static auto test(int)
                -> decltype(nlohmann::adl_serializer<U>::to_json(std::declval<json &>(), std::declval<const U &>()));

            template <class U> static auto test(...) -> No;

            static const bool value = !std::is_same_v<No, decltype(test<T>(0))>;
        };

        template <class T> struct is_json_seriailzable {
            static const bool value =
                is_json_seriailzable_adl_to_json<T>::value || is_json_seriailzable_to_json<T>::value;
        };

        template <class T> struct is_json_deseriailzable {
            struct No {};

            template <class U> static auto test(int) -> decltype(std::declval<json>().get<U>());

            template <class U> static auto test(...) -> No;

            static const bool value = !std::is_same_v<No, decltype(test<T>(0))>;
        };

    } // namespace detail
    class Context {
      public:
    };

    template <class T> struct NVP {
        NVP(const char *name, const T &ref) : name(name), ref(ref) {}

        const char *name;
        const T &ref;
    };

    template <class T> struct NVP_NC {
        NVP_NC(const char *name, T &ref) : name(name), ref(ref) {}

        const char *name;
        T &ref;
    };

    template <class T> struct is_nvp { static const bool value = false; };
    template <class T> struct is_nvp<NVP<T>> { static const bool value = true; };

    template <class T> struct is_nvp<NVP_NC<T>> { static const bool value = true; };

    template <class T> NVP<T> make_nvp(const char *name, const T &arg) { return NVP<T>(name, arg); }

    template <class T> NVP_NC<T> make_nvp(const char *name, T &arg) { return NVP_NC<T>(name, arg); }

    struct ArchiveBase {
        std::vector<std::string> locator;

        template <class F, class... Args> void tryInvoke(F &&f, Args &&... args) {
            try {
                f(std::forward<Args>(args)...);
            } catch (SerializationError &e) {
                throw e;
            } catch (std::exception &e) {
                std::string msg;
                for (auto &i : locator) {
                    msg.append(i);
                }
                throw SerializationError(std::string("error: ").append(e.what()).append(" in ").append(msg));
            }
        }
    };

    class OutputArchive : public ArchiveBase {
        std::vector<std::reference_wrapper<json>> stack;
        std::unordered_map<void *, std::reference_wrapper<json>> ptrs;
        std::vector<int> counter;
        json data;

      public:
        [[nodiscard]] json getData() const { return data; }

        json &_top() { return stack.back(); }

        void _makeNode(json &ref) {
            stack.emplace_back(ref);
            counter.emplace_back(0);
        }

        void _popNode() {
            stack.pop_back();
            counter.pop_back();
        }

        explicit OutputArchive(){ _makeNode(data); }

        template <class T> void _save_nvp(const char *name, const T &value) {
            _top()[name] = json();
            _makeNode(_top()[name]);
            locator.emplace_back(std::string("/").append(name));
            _save(value);
            locator.pop_back();
            _popNode();
        }

        template <class T> void _save_nvp(const std::string &name, const T &value) {
            _top()[name] = json();
            _makeNode(_top()[name]);
            locator.emplace_back(std::string("/").append(name));
            _save(value);
            locator.pop_back();
            _popNode();
        }

        template <class T> void _save(NVP<T> &&nvp) { _save_nvp(nvp.name, nvp.ref); }

        template <typename T> std::enable_if_t<std::is_enum_v<T>> _save(const T &arg) {
            _top() = magic_enum::enum_name<T>(arg);
        }
        template <class T> std::enable_if_t<detail::is_json_seriailzable<T>::value, void> _save(const T &arg) {
            _top() = arg;
        }

        template <class T> std::enable_if_t<detail::has_member_save<T>::value, void> _save(const T &arg) {
            arg.save(*this);
        }

        void _save(const Any &any) {
            if (any.is_shared_pointer()) {
                auto &mgr = akari::detail::reflection_manager::instance();
                auto p = any.__get_internal_shared_pointer();
                auto *raw = p.get();
                if (!raw)
                    return;
                auto it = ptrs.find(raw);
                if (it == ptrs.end()) {
                    _top() = json{{"type", mgr.inv_name_map.at(any.get_underlying_type().name)}};
                    _top()["props"] = json();
                    _makeNode(_top()["props"]);
                    Type type = any.get_underlying_type();
                    type.get_method("save").invoke(any.get_underlying(), make_any_ref(*this));
                    _popNode();
                    ptrs.emplace(raw, _top());
                } else {
                    Type type = any.get_type();
                    type.get_method("save").invoke(any, make_any_ref(*this));
                }
            }
        }
        template <class T> void _save(const std::shared_ptr<T> &ptr) {
            // static_assert(detail::check_serializable_static_type<T>::value, "T must have static Type *
            // staticType()");
            if (!ptr)
                return;
            _save(make_any_ref(ptr));
        }

        template <class T> void _save(const std::vector<T> &vec) {
            _top() = json::array();
            auto &arr = _top();
            auto cnt = 0;
            for (const auto &i : vec) {
                arr.emplace_back();
                _makeNode(arr.back());
                locator.emplace_back(std::string("/").append(std::to_string(cnt)));
                _save(i);
                locator.pop_back();
                _popNode();
                cnt++;
            }
        }

        template <class T> void _save(const std::unordered_map<std::string, T> &map) {
            _top() = json::object();
            auto &dict = _top();
            for (const auto &i : map) {
                dict[i.first] = json();
                _makeNode(dict[i.first]);
                locator.emplace_back(std::string("/").append(i.first));
                _save(i.second);
                locator.pop_back();
                _popNode();
            }
        }

        template <class T> void _save(const NVP<T> &nvp) { _save_nvp(nvp.name, nvp.ref); }

        template <class T> void _save2(T &&arg) {
            if constexpr (is_nvp<T>::value) {
                _save_nvp(arg);
            } else {
                _save_nvp(std::string("$val").append(std::to_string(counter.back())), std::forward<T>(arg));
                counter.back()++;
            }
        }

        template <class T> void _save_v(T &&arg) { _save2(arg); }

        template <class T, class... Args> void _save_v(T &&head, Args &&... args) {
            _save2(head);
            _save_v(args...);
        }

        template <class... Args> void operator()(Args &&... args) { _save_v(args...); }
    };

    class InputArchive : public ArchiveBase {
        const json &data;
        std::vector<std::reference_wrapper<const json>> stack;
        std::unordered_map<size_t, Any> ptrs;
        std::vector<int> counter;

      public:
        const json &_top() { return stack.back(); }

        void _makeNode(const json &ref) {
            stack.emplace_back(ref);
            counter.emplace_back(0);
        }

        void _popNode() {
            stack.pop_back();
            counter.pop_back();
        }

        explicit InputArchive(const json &data) : data(data) {
            stack.emplace_back(data);
        }

        template <class T> std::enable_if_t<detail::is_json_deseriailzable<T>::value, void> _load(T &arg) {
            arg = _top().get<T>();
        }

        template <class T> std::enable_if_t<detail::has_member_load<T>::value, void> _load(T &arg) { arg.load(*this); }
        template <typename T> std::enable_if_t<std::is_enum_v<T>> _save(T &arg) {
            arg = magic_enum::enum_cast<T>(_top().get<std::string>());
        }
        void _load(const Any &any) {
            if (any.is_shared_pointer()) {
                auto type_s = _top().at("type").get<std::string>();
                auto type = Type::get_by_name(type_s.c_str());
                if (_top().contains("props")) {
                    if (_top().contains("addr")) {
                        auto addr = std::stol(_top().at("addr").get<std::string>());
                        auto it = ptrs.find(addr);
                        if (it == ptrs.end()) {
                            any.set_value(type.create_shared());
                            ptrs[addr] = any;
                        } else {
                            any.set_value(it->second);
                        }
                    } else {
                        any.set_value(type.create_shared());
                    }
                    _makeNode(_top().at("props"));
                    type.get_method("load").invoke(any.get_underlying(), make_any_ref(*this));
                    _popNode();
                } else {
                    auto addr = std::stol(_top().at("addr").get<std::string>());
                    any.set_value(ptrs.at(addr));
                }
            } else {
                Type type = any.get_type();
                type.get_method("load").invoke(any, make_any_ref(*this));
            }
        }
        template <class T> void _load(std::shared_ptr<T> &ptr) {
            if (_top().is_null()) {
                ptr = nullptr;
                return;
            }
            _load(make_any_ref(ptr));
        }

        template <class T> void _load(std::vector<T> &vec) {
            for (size_t i = 0; i < _top().size(); i++) {
                _makeNode(_top().at(i));
                locator.emplace_back(std::string("/").append(std::to_string(i)));
                vec.emplace_back();
                _load(vec.back());
                locator.pop_back();
                _popNode();
            }
        }

        template <class T> void _load(std::unordered_map<std::string, T> &map) {
            for (auto &el : _top().items()) {
                _makeNode(el.value());
                locator.emplace_back(std::string("/").append(el.key()));
                map[el.key()] = nullptr;
                _load(map[el.key()]);
                locator.pop_back();
                _popNode();
            }
        }

        template <class T> void _load_nvp(const char *name, T &value) {
            if (_top().contains(name)) {
                _makeNode(_top()[name]);
                locator.emplace_back(std::string("/").append(name));
                _load(value);
                locator.pop_back();
                _popNode();
            }
        }

        template <class T> void _load_nvp(const std::string &name, T &value) {
            if (_top().contains(name)) {
                _makeNode(_top()[name]);
                locator.emplace_back(std::string("/").append(name));
                _load(value);
                locator.pop_back();
                _popNode();
            }
        }

        template <class T> void _load(NVP_NC<T> &&nvp) { _load_nvp(nvp.name, nvp.ref); }

        template <class T> void _load2(T &&arg) {
            if constexpr (is_nvp<T>::value) {
                _load_nvp(arg);
            } else {
                _load_nvp(std::string("$val").append(std::to_string(counter.back())), std::forward<T>(arg));
                counter.back()++;
            }
        }

        template <class T> void _load_v(T &&arg) { _load2(arg); }

        template <class T, class... Args> void _load_v(T &&head, Args &&... args) {
            _load2(head);
            _load_v(args...);
        }

        template <class... Args> void operator()(Args &&... args) { _load_v(args...); }
    };

    struct AutoSaveVisitor {
        OutputArchive &ar;

        template <class T> void visit(T &&arg, const char *name) { ar._save_nvp(name, arg); }
    };

    struct AutoLoadVisitor {
        InputArchive &ar;

        template <class T> void visit(T &&arg, const char *name) { ar._load_nvp(name, arg); }
    };

    template <class T> json save_to_json(const T &data) {
        OutputArchive ar;
        ar.tryInvoke([&]() { ar._save(data); });
        return ar.getData();
    }

    template <class T> T load_from_json(const json &j) {
        InputArchive ar(j);
        T data;
        ar.tryInvoke([&]() { ar._load(data); });
        return data;
    }
    namespace detail {
        template <class Visitor> void accept(Visitor visitor, const char **args_s) {}

        template <class Visitor, class T, class... Args>
        void accept(Visitor visitor, const char **args_s, const T &t, const Args &... args) {
            visitor.visit(t, *args_s);
            accept<Visitor, Args...>(visitor, args_s + 1, args...);
        }

        template <class Visitor, class T, class... Args>
        void accept(Visitor visitor, const char **args_s, T &t, Args &... args) {
            visitor.visit(t, *args_s);
            accept<Visitor, Args...>(visitor, args_s + 1, args...);
        }

        template <class Visitor> void accept(Visitor, const char (&args_s)[1]) {}

        template <class Visitor, size_t N, class... Args>
        void accept(Visitor visitor, const char (&args_s)[N], const Args &... args) {
            std::string s = args_s;
            std::array<std::string, sizeof...(Args)> array;
            size_t pos = 0;
            for (size_t i = 0; i < array.size(); i++) {
                while (pos < s.length() && isspace(s[pos])) {
                    pos++;
                }
                while (pos < s.length() && s[pos] != ',')
                    array[i] += s[pos++];
                pos++;
                while (pos < s.length() && isspace(s[pos])) {
                    pos++;
                }
            }
            const char *a[sizeof...(Args)];
            for (size_t i = 0; i < array.size(); i++) {
                a[i] = array[i].c_str();
            }
            accept<Visitor, Args...>(visitor, a, args...);
        }

        template <class Visitor, size_t N, class... Args>
        void accept(Visitor visitor, const char (&args_s)[N], Args &... args) {
            std::string s = args_s;
            std::array<std::string, sizeof...(Args)> array;
            size_t pos = 0;
            for (size_t i = 0; i < array.size(); i++) {
                while (pos < s.length() && isspace(s[pos])) {
                    pos++;
                }
                while (pos < s.length() && s[pos] != ',')
                    array[i] += s[pos++];
                pos++;
                while (pos < s.length() && isspace(s[pos])) {
                    pos++;
                }
            }
            const char *a[sizeof...(Args)];
            for (size_t i = 0; i < array.size(); i++) {
                a[i] = array[i].c_str();
            }
            accept<Visitor, Args...>(visitor, a, args...);
        }
    } // namespace detail
#define _AKR_DETAIL_REFL(Visitor, ...) akari::serialize::detail::accept(Visitor, #__VA_ARGS__, __VA_ARGS__)

#define AKR_SER(...)                                                                                                   \
     void save(serialize::OutputArchive &ar) const {                                                            \
        akari::serialize::AutoSaveVisitor v{ar};                                                                       \
        _AKR_DETAIL_REFL(v, __VA_ARGS__);                                                                              \
    }                                                                                                                  \
    void load(serialize::InputArchive &ar) {                                                                  \
        akari::serialize::AutoLoadVisitor v{ar};                                                                       \
        _AKR_DETAIL_REFL(v, __VA_ARGS__);                                                                              \
    }
} // namespace akari::Serialize
#endif // AKARIRENDER_SERIALIZE_IMPL_HPP
