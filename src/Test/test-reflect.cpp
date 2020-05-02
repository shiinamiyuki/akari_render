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
#include <akari/core/reflect.hpp>
#include <fmt/format.h>
#include <json.hpp>
struct Foo {
    int a = 123;
    float b = 3.14;
    std::vector<Foo> v;
    Foo() = default;
    Foo(int x, float y) : a(x), b(y) {}
    void hello(int _) const { fmt::print("hello from {}\n", _); }
};
namespace akari {
    using namespace nlohmann;
    struct to_json_impl {
        json data;
        template <typename T> void save(const T &value) {
            Any any(value);
            do_save(data, any);
        }
#define AKR_SAVE_BASIC(ty)                                                                                             \
    if (type_info == type_of<ty>()) {                                                                                  \
        j = any.as<ty>();                                                                                              \
    }
        void do_save(json &j, const Any &any) {
            auto type_info = any.get_type();
            AKR_SAVE_BASIC(int)
            else AKR_SAVE_BASIC(float) else AKR_SAVE_BASIC(double) else {
                if (auto view = any.get_sequential_container_view()) {
                    j = json::array();
                    for (auto iter = view->begin(); iter != view->end(); ++iter) {
                        json tmp;
                        do_save(tmp, *iter);
                        j.push_back(tmp);
                    }
                } else {
                    j = json::object();
                    Type type = any.get_type();
                    if (type.has_method("save")) {
                        type.get_method("save").invoke(any, j);
                    } else {
                        auto props = type.get_properties();
                        for (auto &prop : props) {
                            do_save(j[prop.name()], prop.get(any));
                        }
                    }
                }
            }
        }

#undef AKR_SAVE_BASIC
    };

    template <typename T> json save_to_json(const T &value) {
        to_json_impl impl;
        impl.save(value);
        return std::move(impl.data);
    }
} // namespace akari
int main() {
    using namespace akari;
    // clang-format off
    class_<Foo>("Foo")
        .constructor<>()
            .constructor<int, float>()
            .property("a", &Foo::a)
            .property("b", &Foo::b)
            .property("v", &Foo::v)
            .method("hello",&Foo::hello);
    // clang-format on
    //    Any v = make_any(Foo());
    //    Type type = Type::get<Foo>();
    //    for (auto &prop : v.get_properties()) {
    //        fmt::print("{} \n", prop.second.name());
    //    }

    Type type = Type::get_by_name("Foo");
    auto any = type.create_shared(777, 234.4f);
    for (auto &prop : type.get_properties()) {
        fmt::print("{} \n", prop.name());
    }
    auto *foo = &any.get_underlying().as<Foo>();
    fmt::print("foo.a={}\n", foo->a);
    auto prop = type.get_property("a");
    //    auto any = make_any_ref(foo);
    fmt::print("{}\n", any.is_pointer());
    prop.set(any, 222);
    auto hello = type.get_method("hello");
    hello.invoke(*foo, 23456);
    fmt::print("foo.a={}\n", foo->a);
    {
        Foo foo1;
        foo1.v = {Foo(), Foo()};
        fmt::print("{}\n", save_to_json(foo1).dump(1));
    }
    //    fmt::print("v.a = {}\n", v.get_properties()["a"].as<int>());
}