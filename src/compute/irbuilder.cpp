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

#include <akari/compute/irbuilder.h>
namespace akari::compute::ir {
    class IRBuilderImpl : public IRBuilder {
        std::shared_ptr<FunctionBlock> cur_block;
        Var recent;
      public:
        void set_func_block(std::shared_ptr<FunctionBlock> fb) override { cur_block = std::move(fb); }
        std::shared_ptr<FunctionBlock> get_func_block() override { return cur_block; }
        Var append(const Expr &e) { recent = cur_block->ll.push(e); return recent;}
        Var add_constant(const Constant &c) override { return append(c); }
        Var create_fadd(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::FAdd), lhs, rhs));
        }
        Var create_fsub(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::FSub), lhs, rhs));
        }
        Var create_fmul(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::FMul), lhs, rhs));
        }
        Var create_fdiv(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::FDiv), lhs, rhs));
        }
        Var create_iadd(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::IAdd), lhs, rhs));
        }
        Var create_isub(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::ISub), lhs, rhs));
        }
        Var create_imul(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::IMul), lhs, rhs));
        }
        Var create_idiv(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::IDiv), lhs, rhs));
        }
        Var create_imod(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::IMod), lhs, rhs));
        }
        Var create_and(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::And), lhs, rhs));
        }
        Var create_or(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::And), lhs, rhs));
        }
        Var create_xor(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::Xor), lhs, rhs));
        }
        Var create_shl(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::Shl), lhs, rhs));
        }
        Var create_shr(const Var &lhs, const Var &rhs) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::Shr), lhs, rhs));
        }
        Var create_not(const Var &v) override {
            return append(std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::Not), v));
        }
        Var create_fneg(const Var &v) override {
            return append(std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::FNeg), v));
        }
        Var create_ineg(const Var &v) override {
            return append(std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::INeg), v));
        }

        Var create_sp2i(const Var &v) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::ConvertSP2I), v));
        };
        Var create_dp2i(const Var &v) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::ConvertDP2I), v));
        };
        Var create_i2sp(const Var &v) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::ConvertI2SP), v));
        };
        Var create_i2dp(const Var &v) override {
            return append(
                std::make_shared<CallNode>(std::make_shared<PrimitiveNode>(PrimitiveOp::ConvertI2DP), v));
        };
        void create_ret() override {
            cur_block->ret = recent;
        }
    };

    thread_local IRBuilderImpl ir_builder;

    IRBuilder *IRBuilder::get() { return &ir_builder; }
} // namespace akari::compute::ir
