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
#include <akari/common/math.h>
#include <pybind11/pybind11.h>
#include <akari/core/logger.h>
namespace akari {
    namespace py = pybind11;
    template <typename T> struct RegisterArrayOp {
        template <typename Class> void operator()(py::module &m, Class &c) const {
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
            c.def("__str__", [](const T &a) -> std::string { return fmt::format("{}", a); });
            m.def("degrees", [](const T &a) { return degrees(a); });
            m.def("radians", [](const T &a) { return radians(a); });
            // if constexpr (is_integer_v<T>){

            // }
        }
    };

    AKR_VARIANT void register_math_functions(py::module &m) {
        AKR_IMPORT_CORE_TYPES()

        {
            auto c = py::class_<Vector2f>(m, "Vector2f")
                         .def(py::init<>())
                         .def(py::init<Float>())
                         .def(py::init<Float, Float>());
            RegisterArrayOp<Vector2f>()(m, c);
        }
        {
            auto c = py::class_<Vector3f>(m, "Vector3f")
                         .def(py::init<>())
                         .def(py::init<Float>())
                         .def(py::init<Float, Float>())
                         .def(py::init<Float, Float, Float>());
            RegisterArrayOp<Vector3f>()(m, c);
        }
        {
            auto c = py::class_<Point2i>(m, "Point2i")
                         .def(py::init<>())
                         .def(py::init<Int32>())
                         .def(py::init<Int32, Int32>());
            RegisterArrayOp<Point2i>()(m, c);
        }
        {
            auto c = py::class_<Point2f>(m, "Point2f")
                         .def(py::init<>())
                         .def(py::init<Float>())
                         .def(py::init<Float, Float>());
            RegisterArrayOp<Point2f>()(m, c);
        }
        {
            auto c = py::class_<Point3f>(m, "Point3f")
                         .def(py::init<>())
                         .def(py::init<Float>())
                         .def(py::init<Float, Float>())
                         .def(py::init<Float, Float, Float>());
            RegisterArrayOp<Point3f>()(m, c);
        }
    }
} // namespace akari