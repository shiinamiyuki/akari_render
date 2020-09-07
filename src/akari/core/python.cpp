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

#include <akari/common/fwd.h>
#include <akari/core/python.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <akari/core/scenegraph.h>
namespace akari {
    template <typename Float, typename T>
    struct RegisterArrayOp {
        template <typename Class>
        void operator()(py::module &m, Class &c) const {
            AKR_IMPORT_CORE_TYPES()
            constexpr auto N = array_size_v<T>;
            if constexpr (!std::is_same_v<bool, value_t<T>>) {
                c.def("__add__", [](const T &a, const T &b) { return a + b; });
                c.def("__sub__", [](const T &a, const T &b) { return a - b; });
                c.def("__mul__", [](const T &a, const T &b) { return a * b; });
                c.def("__truediv__", [](const T &a, const T &b) { return a / b; });
                c.def("__add__", [](const T &a, const value_t<T> &b) { return a + b; });
                c.def("__sub__", [](const T &a, const value_t<T> &b) { return a - b; });
                c.def("__mul__", [](const T &a, const value_t<T> &b) { return a * b; });
                c.def("__truediv__", [](const T &a, const value_t<T> &b) { return a / b; });
                c.def("__add__", [](const value_t<T> &a, const T &b) { return a + b; });
                c.def("__sub__", [](const value_t<T> &a, const T &b) { return a - b; });
                c.def("__mul__", [](const value_t<T> &a, const T &b) { return a * b; });
                c.def("__truediv__", [](const value_t<T> &a, const T &b) { return a / b; });
#define REGISTER_MATH_OP(name, op) c.def(#name, [](const T &a, const T &b) { return a op b; });
                REGISTER_MATH_OP("__lt__", <)
                REGISTER_MATH_OP("__gt__", >)
                REGISTER_MATH_OP("__le__", <=)
                REGISTER_MATH_OP("__ge__", >=)
                REGISTER_MATH_OP("__eq__", ==)
                REGISTER_MATH_OP("__ne__", !=)
            }
            c.def("__str__", [](const T &a) -> std::string { return fmt::format("{}", a); });
            if constexpr (!std::is_same_v<bool, value_t<T>>) {
                m.def("degrees", [](const T &a) { return degrees(a); });
                m.def("radians", [](const T &a) { return radians(a); });
                m.def("dot", [](const T &u, const T &v) -> T { return dot(u, v); });
#define REGISTER_MATH_FUNC2(name)                                                                                      \
    m.def(#name, [](const T &u, const T &v) -> T { return name(u, v); });                                              \
    m.def(#name, [](const value_t<T> &u, const T &v) -> T { return name(u, v); });                                     \
    m.def(#name, [](const T &u, const value_t<T> &v) -> T { return name(u, v); });
                REGISTER_MATH_FUNC2(min)
                REGISTER_MATH_FUNC2(max)
                REGISTER_MATH_FUNC2(pow)
                m.def("clamp", [](const T &x, const T &a, const T &b) -> T { return clamp(x, a, b); });
            }
            c.def(py::init<Array<int, N>>());
            c.def(py::init<Array<float, N>>());
            c.def(py::init<Array<double, N>>());
            c.def(py::init<Array<bool, N>>());
            c.def(py::init<>());
            if constexpr (N >= 1) {
                c.def(py::init<value_t<T>>());
            }
            if constexpr (N >= 2) {
                c.def(py::init<value_t<T>, value_t<T>>());
            }
            if constexpr (N >= 3) {
                c.def(py::init<value_t<T>, value_t<T>, value_t<T>>());
            }
        }
    };
    AKR_VARIANT void RegisterMathFunction<C>::register_math_functions(py::module &m) {
        AKR_IMPORT_TYPES()
        m.def("degrees", [](const Float &a) { return degrees(a); });
        m.def("radians", [](const Float &a) { return radians(a); });
        {
            auto c = py::class_<Vector2f>(m, "Vector2f");
            RegisterArrayOp<Float, Vector2f>()(m, c);
        }
        {
            auto c = py::class_<Vector3f>(m, "Vector3f");
            RegisterArrayOp<Float, Vector3f>()(m, c);
        }
        {
            auto c = py::class_<Point2i>(m, "Point2i");
            RegisterArrayOp<Float, Point2i>()(m, c);
        }
        {
            auto c = py::class_<Point2f>(m, "Point2f");
            RegisterArrayOp<Float, Point2f>()(m, c);
        }
        {
            auto c = py::class_<Point3f>(m, "Point3f");
            RegisterArrayOp<Float, Point3f>()(m, c);
        }
        {
            auto c = py::class_<Point3i>(m, "Point3i");
            RegisterArrayOp<Float, Point3i>()(m, c);
        }
        {
            auto c = py::class_<Array2b>(m, "Array2b");
            RegisterArrayOp<Float, Array2b>()(m, c);
        }
        {
            auto c = py::class_<Array3b>(m, "Array3b");
            RegisterArrayOp<Float, Array3b>()(m, c);
        }
        {
            auto c = py::class_<Array4b>(m, "Array4b");
            RegisterArrayOp<Float, Array4b>()(m, c);
        }
        {
            auto c = py::class_<Array2f>(m, "Array2f");
            RegisterArrayOp<Float, Array2f>()(m, c);
        }
        {
            auto c = py::class_<Array3f>(m, "Array3f");
            RegisterArrayOp<Float, Array3f>()(m, c);
        }
        {
            auto c = py::class_<Array4f>(m, "Array4f");
            RegisterArrayOp<Float, Array4f>()(m, c);
        }
        {
            auto c = py::class_<Array2i>(m, "Array2i");
            RegisterArrayOp<Float, Array2i>()(m, c);
        }
        {
            auto c = py::class_<Array3i>(m, "Array3i");
            RegisterArrayOp<Float, Array3i>()(m, c);
        }
        {
            auto c = py::class_<Array4i>(m, "Array4i");
            RegisterArrayOp<Float, Array4i>()(m, c);
        }
        {
            auto c = py::class_<Color3f>(m, "Color3f");
            RegisterArrayOp<Float, Color3f>()(m, c);
        }
    }
    AKR_RENDER_STRUCT(RegisterMathFunction)

    static py::object exported;
    py::object load_fragment(const std::string &path) {
        py::eval_file(py::str(path));
        auto ret = exported;
        exported = py::none();
        return ret;
    }
    void py_export(py::object obj) { exported = obj; }
    void register_utility(py::module &m) {
        m.def("load_fragment", &load_fragment);
        m.def("export", &py_export);
    }
    std::vector<const char *> _get_enabled_variants() {
        return std::vector<const char *>(std::begin(enabled_variants), std::end(enabled_variants));
    }
    AKR_VARIANT void register_scene_graph(py::module &m) { RegisterSceneGraph<C>::register_scene_graph(m); }
    PYBIND11_EMBEDDED_MODULE(akari, m) {
        register_utility(m);
        m.def("enabled_variants", _get_enabled_variants);
        for (auto &variant : enabled_variants) {
            AKR_INVOKE_VARIANT(std::string_view(variant), register_scene_graph, m);
        }
    }
} // namespace akari