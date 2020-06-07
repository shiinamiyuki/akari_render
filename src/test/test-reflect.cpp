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

#include <json.hpp>
#include <akari/core/detail/serialize-impl.hpp>
#include <akari/core/math.h>
#include <akari/core/reflect.hpp>
#include <akari/core/serialize.hpp>
#include <fmt/format.h>


using namespace akari;
using namespace akari::serialize;
struct Object {
    virtual ~Object()=default;
};
struct Foo : Object {
    int a = 123;
    float b = 3.14;
    glm::vec3 u;
    std::vector<Foo> v;

    Foo() = default;

    Foo(int x, float y) : a(x), b(y) {}

    void hello(int _) const { fmt::print("hello from {}\n", _); }

    void save(OutputArchive &ar) { ar(a, b, u, v); }
    template<typename T>
    void foo(){};
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
            auto type_info = any._get_type();
            AKR_SAVE_BASIC(int)
            else AKR_SAVE_BASIC(float) else AKR_SAVE_BASIC(double) else {
                if (auto view = any.get_sequential_container_view()) {
                    j = json::array();
                    for (auto iter = view->begin(); iter != view->end(); ++iter) {
                        json tmp;
                        do_save(tmp, *iter);
                        j.push_back(tmp);
                    }
                } else if (auto opt = dynamic_invoke_noexcept("to_json", make_any_ref(j), any)) {
                    (void)opt;
                } else {
                    j = json::object();
                    Type type = any._get_type();
                    if (type.has_method("save")) {
                        type.get_method("save").invoke(any, make_any_ref(j));
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

    template <typename T> struct json_serializer {
        static void save(json &j, const T &value) {
            printf("called to_json %p\n", &j);
            to_json(j, value);
        }
    };
} // namespace akari

struct Base {
    virtual void f() = 0;
};

struct Derived : Base {
    int i = 0;
    void f() override {
        i++;
        printf("derived %d\n", i);
    }
};
void good(std::shared_ptr<Base> p){
    p->f();
}
int main() {

    // clang-format off
    class_<Foo, Object>("Foo")
            .constructor<>()
            .constructor<int, float>()
            .property("a", &Foo::a)
            .property("b", &Foo::b)
            .property("v", &Foo::v)
            .property("u", &Foo::u)
            .method("hello", &Foo::hello)
            .method("foo",&Foo::foo<int>)
            .method("save",&Foo::save);
    function("to_json", &json_serializer<vec3>::save);
    class_<Derived,Base>("Derived")
            .constructor<>();
    // clang-format on
    //    Any v = make_any(Foo());
    //    Type type = Type::get<Foo>();
    //    for (auto &prop : v.get_properties()) {
    //        fmt::print("{} \n", prop.second.name());
    //    }
//    {
//        Type type = Type::get_by_name("Foo");
//        auto any = type.create_shared(777, 234.4f);
//        for (auto &prop : type.get_properties()) {
//            fmt::print("{} \n", prop.name());
//        }
//        auto *foo = &any.get_underlying().as<Foo>();
//        fmt::print("foo.a={}\n", foo->a);
//        auto prop = type.get_property("a");
//        //    auto any = make_any_ref(foo);
//        fmt::print("{}\n", any.is_pointer());
//        prop.set(any, 222);
//        auto hello = type.get_method("hello");
//        hello.invoke(*foo, 23456);
//        fmt::print("foo.a={}\n", foo->a);
//        {
//            json _j;
//            Foo foo1;
//            printf("%p\n", &_j);
//            foo1.u = vec3(1, 2, 3);
//            dynamic_invoke("to_json", _j, foo1.u);
//            foo1.v = {Foo(), Foo()};
//            fmt::print("{}\n", save_to_json(foo1).dump(1));
//        }
//    }
    //    fmt::print("v.a = {}\n", v.get_properties()["a"].as<int>());
    {
        auto derived_t = Type::get_by_name("Derived");
        auto d = derived_t.create_shared();
        auto derived = d.shared_cast<Base>();
        derived->f();
        auto base = make_any_ref(derived).shared_cast<Derived>();
        base->f();
    }
    {
        std::shared_ptr<Foo> foo(new Foo());
        foo->a = 8888;
        OutputArchive ar;
        ar(foo);
        fmt::print("{}\n",ar.getData().dump(1));
    }
    {
        Function f(&good);
        f.invoke(Any(std::make_shared<Derived>()));
    }
    {
        struct Object{};
        class_<Object>("Object")
            .constructor<>()
            .method("foo",std::function<void(Object&)>([](Object&){printf("hello\n");}));
    }
    {
        auto t = Type::get_by_name("Object");
        auto object = t.create();
        t.get_method("foo").invoke(object);
    }
}