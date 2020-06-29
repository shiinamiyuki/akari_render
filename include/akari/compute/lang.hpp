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
#include <variant>
#include <magic_enum.hpp>
#include <unordered_set>
#include <iostream>
#include <functional>

#include <akari/compute/ir.hpp>
#include <akari/compute/letlist.hpp>
#include <akari/compute/irbuilder.h>

// AkariCompute Embedded DSL
namespace akari::compute::lang {
    template <typename T> ir::Type get_type_from_native() {
        if constexpr (std::is_same_v<T, float>) {
            return ir::get_primitive_type(ir::PrimitiveTy::float32);
        } else if constexpr (std::is_same_v<T, double>) {
            return ir::get_primitive_type(ir::PrimitiveTy::float64);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return ir::get_primitive_type(ir::PrimitiveTy::int32);
        } else if constexpr (std::is_same_v<T, bool>) {
            return ir::get_primitive_type(ir::PrimitiveTy::boolean);
        } else {
            return nullptr;
        }
    }

    // class Thunk {
    // public:
    //     Thunk(const Thunk&)=delete;
    //     Thunk(Thunk&&)=delete;
    //     Thunk&operator=(const Thunk&)=delete;
    //     Thunk&operaotr=(Thunk&&)=delete;
    // };

    template <typename T> struct Var {
        template <typename U> Var(const Var<U> &rhs) {
            using From = U;
            using To = T;
            if constexpr (std::is_same_v<From, int32_t>) {
                if constexpr (std::is_same_v<To, float>) {
                    var = ir::IRBuilder::get()->create_i2sp(rhs);
                } else if constexpr (std::is_same_v<To, double>) {
                    var = ir::IRBuilder::get()->create_i2dp(rhs);
                } else {
                    static_assert(false);
                }
            } else if constexpr (std::is_same_v<From, float>) {
                if constexpr (std::is_same_v<To, int32_t>) {
                    var = ir::IRBuilder::get()->create_sp2i(rhs);
                } else {
                    static_assert(false);
                }
            } else if constexpr (std::is_same_v<From, double>) {
                if constexpr (std::is_same_v<To, int32_t>) {
                    var = ir::IRBuilder::get()->create_dp2i(rhs);
                } else {
                    static_assert(false);
                }
            }
        }
        Var(const T &v) { var = ir::IRBuilder::get()->add_constant(std::make_shared<ir::ConstantNode>(v)); }
        template <typename U> auto operator+(const Var<U> &rhs) const {
            using R = decltype(std::declval<T>() + std::declval<U>());
            Var<R> converted_lhs = *this, converted_rhs = rhs;
            Var<R> res;
            if constexpr (std::is_same_v < std::is_floating_point_v<R>) {
                res.var = ir::IRBuilder::get()->create_fadd(converted_lhs.var,converted_rhs.var);
            } else {
                res.var = ir::IRBuilder::get()->create_iadd(converted_lhs.var,converted_rhs.var);
            }
            return res;
        }
        template <typename U> auto operator-(const Var<U> &rhs) const {
            using R = decltype(std::declval<T>() - std::declval<U>());
            Var<R> converted_lhs = *this, converted_rhs = rhs;
            Var<R> res;
            if constexpr (std::is_same_v < std::is_floating_point_v<R>) {
                res.var = ir::IRBuilder::get()->create_fsub(converted_lhs.var,converted_rhs.var);
            } else {
                res.var = ir::IRBuilder::get()->create_isub(converted_lhs.var,converted_rhs.var);
            }
            return res;
        }

        template <typename U> auto operator*(const Var<U> &rhs) const {
            using R = decltype(std::declval<T>() * std::declval<U>());
            Var<R> converted_lhs = *this, converted_rhs = rhs;
            Var<R> res;
             if constexpr (std::is_same_v < std::is_floating_point_v<R>) {
                res.var = ir::IRBuilder::get()->create_fmul(converted_lhs.var,converted_rhs.var);
            } else {
                res.var = ir::IRBuilder::get()->create_imul(converted_lhs.var,converted_rhs.var);
            }
            return res;
        }

        template <typename U> auto operator/(const Var<U> &rhs) const {
            using R = decltype(std::declval<T>() / std::declval<U>());
            Var<R> converted_lhs = *this, converted_rhs = rhs;
            Var<R> res;
            if constexpr (std::is_same_v < std::is_floating_point_v<R>) {
                res.var = ir::IRBuilder::get()->create_fdiv(converted_lhs.var,converted_rhs.var);
            } else {
                res.var = ir::IRBuilder::get()->create_idiv(converted_lhs.var,converted_rhs.var);
            }
            return res;
        }
        template <typename U> auto operator%(const Var<U> &rhs) const {
            static_assert(std::is_integral_v<T> && std::is_integral_v<U>);
            using R = decltype(std::declval<T>() % std::declval<U>());
            Var<R> converted_lhs = *this, converted_rhs = rhs;
            Var<R> res;
            res.var = ir::IRBuilder::get()->create_imod(converted_lhs.var,converted_rhs.var);
            return res;
        }
        template <typename U> auto operator<<(const Var<U> &rhs) const {
            static_assert(std::is_integral_v<T> && std::is_integral_v<U>);
            using R = decltype(std::declval<T>() << std::declval<U>());
            Var<R> converted_lhs = *this, converted_rhs = rhs;
            Var<R> res;
            res.var = ir::IRBuilder::get()->create_shl(converted_lhs.var,converted_rhs.var);
            return res;
        }
        template <typename U> auto operator>>(const Var<U> &rhs) const {
            static_assert(std::is_integral_v<T> && std::is_integral_v<U>);
            using R = decltype(std::declval<T>() >> std::declval<U>());
            Var<R> converted_lhs = *this, converted_rhs = rhs;
            Var<R> res;
            res.var = ir::IRBuilder::get()->create_shr(converted_lhs.var,converted_rhs.var);
            return res;
        }
        template <typename U> Var &operator+=(const Var<U> &rhs) {
            *this = *this + rhs;
            return *this;
        }
        template <typename U> Var &operator-=(const Var<U> &rhs) {
            *this = *this - rhs;
            return *this;
        }
        template <typename U> Var &operator*=(const Var<U> &rhs) {
            *this = *this * rhs;
            return *this;
        }
        template <typename U> Var &operator/=(const Var<U> &rhs) {
            *this = *this / rhs;
            return *this;
        }
        template <typename U> Var &operator%=(const Var<U> &rhs) {
            *this = *this % rhs;
            return *this;
        }

      private:
        ir::Var var;
    };

    // template <typename Ret, typename... Args> struct Function {
    //     Function(Ret (*func)(Args &&...)) {
            
    //     }

    //   private:
    //     ir::Function function;
    // };

    using boolean = Var<bool>;
    using int32 = Var<int32_t>;
    using float32 = Var<float>;
    using float64 = Var<double>;
} // namespace akari::compute::lang