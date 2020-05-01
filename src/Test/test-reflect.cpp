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
#include <Akari/Core/Reflect.hpp>
#include <fmt/format.h>

struct Foo {
    int a = 123;
    float b = 3.14;
    Foo() = default;
    Foo(int x, float y) : a(x), b(y) {}
};
int main() {
    using namespace Akari;
    register_type<Foo>("Foo").constructor<>().constructor<int, float>().property("a", &Foo::a).property("b", &Foo::b);
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
    fmt::print("foo.a={}\n", foo->a);
    //    fmt::print("v.a = {}\n", v.get_properties()["a"].as<int>());
}