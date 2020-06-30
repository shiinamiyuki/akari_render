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
#include <stack>
#include <akari/compute/ir.hpp>
#include <akari/compute/letlist.hpp>
namespace akari::compute::ir {

    class FunctionBlock {
      public:
        std::vector<Var> parameters;
        LetList ll;
        Expr ret;
        Function get_func_node() { return std::make_shared<FunctionNode>(parameters, ll.get(ret)); }
    };
    class FunctionStack {
        std::stack<std::shared_ptr<FunctionBlock>> st;

      public:
        void push_func_block(std::shared_ptr<FunctionBlock> fb) { st.emplace(fb); }
        std::shared_ptr<FunctionBlock> pop_func_block() {
            auto fb = st.top();
            st.pop();
            return fb;
        }
    };
    class AKR_EXPORT IRBuilder {
      public:
        virtual void set_func_block(std::shared_ptr<FunctionBlock> fb) = 0;
        virtual std::shared_ptr<FunctionBlock> get_func_block() = 0;
        virtual Var add_constant(const Constant &) = 0;
        virtual Var create_fadd(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_fsub(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_fmul(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_fdiv(const Var &lhs, const Var &rhs) = 0;
        virtual void create_ret() = 0;

        virtual Var create_iadd(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_isub(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_imul(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_idiv(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_imod(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_and(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_or(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_xor(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_shl(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_shr(const Var &lhs, const Var &rhs) = 0;
        virtual Var create_not(const Var &v) = 0;
        virtual Var create_fneg(const Var &v) = 0;
        virtual Var create_ineg(const Var &v) = 0;
        virtual Var create_sp2i(const Var &v) = 0;
        virtual Var create_dp2i(const Var &v) = 0;
        virtual Var create_i2sp(const Var &v) = 0;
        virtual Var create_i2dp(const Var &v) = 0;
        virtual Var make_parameter(const Type & ty) =0 ;
        static IRBuilder *get();
    };
} // namespace akari::compute::ir