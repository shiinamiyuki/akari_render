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

#include <type_traits>
#include <string_view>
#include <akari/common/detail/macro.h>

namespace akari {
#define AKR_STRUCT_REFL_MEMBER(member)                                                                                 \
    {                                                                                                                  \
        auto get_member = [](auto &object) -> auto & { return object.member; };                                        \
        auto get_member_cst = [](const auto &object) -> const auto & { return object.member; };                        \
        vis(get_member, get_member_cst);                                                                               \
    }
#define AKR_STRUCT_REFL_0(x, peek, ...)                                                                                \
    AKR_EVAL_0(AKR_STRUCT_REFL_MEMBER(x))                                                                              \
    AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_1)(peek, __VA_ARGS__)
#define AKR_STRUCT_REFL_1(x, peek, ...)                                                                                \
    AKR_EVAL_0(AKR_STRUCT_REFL_MEMBER(x))                                                                              \
    AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_0)(peek, __VA_ARGS__)
#define AKR_STRUCT_REFL_2(peek, ...) AKR_EVAL(AKR_MAP_STMT_NEXT(peek, AKR_STRUCT_REFL_0)(peek, __VA_ARGS__))

#define AKR_STRUCT_REFL(...)                                                                                           \
    template <class Visitor>                                                                                           \
    AKR_XPU static void _for_each_field(Visitor &&vis) {                                                               \
        AKR_EVAL_0(AKR_STRUCT_REFL_2(__VA_ARGS__, (), 0))                                                              \
    }                                                                                                                  \
    \ template <class Visitor>                                                                                         \
    static void _for_each_field_cpu(Visitor &&vis) {                                                                   \
        AKR_EVAL_0(AKR_STRUCT_REFL_2(__VA_ARGS__, (), 0))                                                              \
    }
    template <typename T, class Visitor>
    inline void unfold_fields(T &v, Visitor &&vis) {
        if constexpr (std::is_fundamental_v<T>) {
            vis(v);
        } else {
            T::foreach_fields([&](auto &&get) { unfold_fields(get(v), vis); });
        }
    }
    template <typename T, class Visitor>
    inline void unfold_fields(const T &v, Visitor &&vis) {
        if constexpr (std::is_fundamental_v<T>) {
            vis(v);
        } else {
            T::foreach_fields([&](auto &&get) { unfold_fields(get(v), vis); });
        }
    }
    template <typename T, class Visitor>
    AKR_XPU inline void foreach_field_(const T &v, Visitor &&vis) {
        T::_for_each_field_cpu([&](auto &&get, auto &&get_c) { vis(get_c); });
    }
    template <typename T, class Visitor>
    AKR_XPU inline void foreach_field(T &v, Visitor &&vis) {
        T::_for_each_field_cpu([&](auto &&get, auto &&get_c) { vis(get); });
    }
    template <typename T, class Visitor>
    inline void foreach_field_cpu(const T &v, Visitor &&vis) {
        T::_for_each_field_cpu([&](auto &&get, auto &&get_c) { vis(get_c); });
    }
    template <typename T, class Visitor>
    inline void foreach_field_cpu(T &v, Visitor &&vis) {
        T::_for_each_field_cpu([&](auto &&get, auto &&get_c) { vis(get); });
    }

} // namespace akari