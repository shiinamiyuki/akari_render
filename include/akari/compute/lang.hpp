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

// AkariCompute Embedded DSL
namespace akari::compute::lang {
    static size_t new_var_id() {
        static size_t id = 0;
        return id++;
    }

    // template<typename T=void>
    namespace detail {
        struct named {};

        struct any {};
    } // namespace detail

    struct Function;
    namespace type {
        struct Generic {};
        template <typename T, size_t... Dims> struct Tensor {};
        template <typename T> using Scalar = Tensor<T, 1>;
        template <typename T, size_t N> using Vector = Tensor<T, N>;
        template <typename T, size_t M, size_t N> using Matrix = Tensor<T, M, N>;

    } // namespace type
    struct Var {
        template <typename T, typename = std::enable_if_t<ir::is_value<T>::value>>
        Var(const T &v, const std::string &name = "") : name(name) {
            expr = std::make_shared<ir::ConstantNode>(v);
        }
        Var() = default;
        inline Var(const Function &f);
        Var(detail::named, const std::string &name = "") { expr = std::make_shared<ir::VarNode>(new_var_id(), name); }
        Var(std::shared_ptr<ir::Expr> expr) : expr(std::move(expr)) {}
        const std::shared_ptr<ir::Expr> &get_expr() const { return expr; }
        friend Var operator+(const Var &lhs, const Var &rhs) {
            return Var(ir::call(ir::Primitive::EAdd, lhs.expr, rhs.expr));
        }
        friend Var operator-(const Var &lhs, const Var &rhs) {
            return Var(ir::call(ir::Primitive::ESub, lhs.expr, rhs.expr));
        }
        friend Var operator*(const Var &lhs, const Var &rhs) {
            return Var(ir::call(ir::Primitive::EMul, lhs.expr, rhs.expr));
        }
        friend Var operator/(const Var &lhs, const Var &rhs) {
            return Var(ir::call(ir::Primitive::EDiv, lhs.expr, rhs.expr));
        }
        Var operator+=(const Var &rhs) {
            *this = *this + rhs;
            return *this;
        }
        Var operator-=(const Var &rhs) {
            *this = *this - rhs;
            return *this;
        }
        Var operator*=(const Var &rhs) {
            *this = *this * rhs;
            return *this;
        }
        Var operator/=(const Var &rhs) {
            *this = *this / rhs;
            return *this;
        }

      protected:
        std::string name;
        std::shared_ptr<ir::Expr> expr;
    };

    struct Function : Var{
        template <typename T, T... ints, typename F>
        Var invoke(std::index_sequence<ints...>, const F &f, const std::vector<Var> &parameters) {
            return f(parameters.at(ints)...);
        }
        template <typename... Ts>
        // typename = std::enable_if_t<std::conjunction_v<std::is_same_v<std::decay_t<Ts>, Var>>>>
        Function(std::function<Var(Ts...)> f) {
            std::vector<Var> parameters;
            for (size_t i = 0; i < sizeof...(Ts); i++) {
                parameters.emplace_back(detail::named{}, fmt::format("param{}", i));
            }

            Var body(invoke<size_t>(std::index_sequence_for<Ts...>{}, f, parameters));
            std::vector<std::shared_ptr<ir::VarNode>> true_parameters;
            std::transform(parameters.begin(), parameters.end(), std::back_inserter(true_parameters),
                           [](const Var &var) {
                               AKR_ASSERT(var.get_expr() && var.get_expr()->isa<ir::VarNode>());
                               return var.get_expr()->cast<ir::VarNode>();
                           });
            // AKR_ASSERT(body.get_expr());
            expr = std::make_shared<ir::FunctionNode>(true_parameters, body.get_expr());
        }
        template <typename... Ts> Var operator()(Ts &&... args) {
            std::vector<Var> parameters{args...};
            std::vector<ir::NodePtr> true_parameters;
            for (auto &p : parameters) {
                true_parameters.emplace_back(p.get_expr());
            }
            return Var(std::make_shared<ir::CallNode>(expr, true_parameters));
        }
        template <typename F> explicit Function(F &&f) : Function(std::function(f)) {}
    };
    inline Var::Var(const Function &f) : expr(f.expr) {}
} // namespace akari::compute::lang