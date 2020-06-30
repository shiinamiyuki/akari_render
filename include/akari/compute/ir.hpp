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
#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <akari/core/platform.h>
#include <akari/compute/base.hpp>

namespace akari::compute::ir {
    class TypeNode : public Base {};
#define AKR_DECL_TYPENODE(Type)                                                                                        \
    std::string type_name() const { return #Type; }
    enum class PrimitiveTy {
        boolean,
        int32,
        float32,
        float64,
    };
    using Type = std::shared_ptr<TypeNode>;
    class PrimitiveTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(PrimitiveTypeNode)
        PrimitiveTypeNode(PrimitiveTy prim) : prim(prim) {}
        PrimitiveTy prim;
    };
    using PrimitiveType = std::shared_ptr<PrimitiveTypeNode>;
    AKR_EXPORT PrimitiveType get_primitive_type(PrimitiveTy);

    AKR_EXPORT size_t generate_id();
    class NodeVisitor;
    class CallNode;
    class ConstantNode;
    class FunctionNode;
    class PrimitiveNode;
    class IfNode;
    class LetNode;
    class VarNode;
    class NodeVisitor {
      public:
        virtual void visit(CallNode &) {}
        virtual void visit(ConstantNode &) {}
        virtual void visit(FunctionNode &) {}
        virtual void visit(PrimitiveNode &) {}
        virtual void visit(LetNode &) {}
        virtual void visit(IfNode &) {}
        virtual void visit(VarNode &) {}
    };
    class IRNode : public Base {
      public:
        Type type;
        virtual void accept(NodeVisitor &) = 0;
    };
    class AtomicNode : public IRNode {};
    using Atomic = std::shared_ptr<AtomicNode>;
    using Node = std::shared_ptr<IRNode>;
    using Var = std::shared_ptr<VarNode>;
    using Let = std::shared_ptr<LetNode>;
    using Call = std::shared_ptr<CallNode>;
    using Function = std::shared_ptr<FunctionNode>;
    using Constant = std::shared_ptr<ConstantNode>;
#define AKR_DECL_NODE(Type)                                                                                            \
    std::string type_name() const { return #Type; }                                                                    \
    void accept(NodeVisitor &vis) { vis.visit(*this); }
    // class Expr : public Node {};
    using ExprNode = IRNode;
    using Expr = std::shared_ptr<ExprNode>;
    class VarNode : public AtomicNode {
        size_t id;
        std::string _name;

      public:
        explicit VarNode(size_t id) : id(id), _name(std::to_string(id)) {}
        VarNode(size_t id, std::string name) : id(id), _name(std::move(name)) {}
        auto &name() const { return _name; }
        AKR_DECL_NODE(VarNode)
    };

    enum class PrimitiveOp {
        Invalid,
        // GenericBegin,
        // GAdd,
        // GSub,
        // GMul,
        // GDiv,
        // GMod,
        // GenericEnd,
        INeg,
        IAdd,
        ISub,
        IMul,
        IDiv,
        IMod,
        Shl,
        Shr,
        And,
        Or,
        Xor,
        Not,
        FNeg,
        FAdd,
        FSub,
        FMul,
        FDiv,
        ConvertSP2I,
        ConvertI2SP,
        ConvertDP2I,
        ConvertI2DP,
    };

    class PrimitiveNode : public AtomicNode {
        PrimitiveOp _prim;

      public:
        PrimitiveNode(PrimitiveOp p) : _prim(p) {}
        PrimitiveOp primitive() const { return _prim; }
        AKR_DECL_NODE(PrimitiveNode)
    };
    using Primitive = std::shared_ptr<PrimitiveNode>;
    // Anonymous Function Node
    class FunctionNode : public ExprNode {
        std::vector<Var> _parameters;
        Expr _body;

      public:
        FunctionNode(std::vector<Var> parameters, Expr body)
            : _parameters(std::move(parameters)), _body(std::move(body)) {}
        const std::vector<Var> &parameters() const { return _parameters; }
        auto body() const { return _body; }
        AKR_DECL_NODE(FunctionNode)
    };
    class IfNode : public ExprNode {
        Node _cond, _then, _else;

      public:
        IfNode(Node _cond, Node _then, Node _else)
            : _cond(std::move(_cond)), _then(std::move(_then)), _else(std::move(_else)) {}
        AKR_DECL_NODE(IfNode)
        const Node &cond() const { return _cond; }
        const Node &then() const { return _then; }
        const Node &else_() const { return _else; }
    };
    using If = std::shared_ptr<IfNode>;
    class LetNode : public ExprNode {
        Var _var;
        Node _val, _body;

      public:
      public:
        LetNode(Var _var, Node _val, Node _body)
            : _var(std::move(_var)), _val(std::move(_val)), _body(std::move(_body)) {}
        const Var &var() const { return _var; }
        const Node &value() const { return _val; }
        const Node &body() const { return _body; }
        AKR_DECL_NODE(LetNode)
    };
    template <typename T> struct is_value : std::false_type {};
    template <> struct is_value<int> : std::true_type {};
    template <> struct is_value<float> : std::true_type {};

    class CallNode : public ExprNode {
        Atomic _op;
        std::vector<Atomic> _args;

      public:
        CallNode() = default;
        CallNode(Atomic op, const Atomic &a0) : _op(std::move(op)), _args({a0}) {}
        CallNode(Atomic op, const Atomic &a0, const Atomic &a1) : _op(std::move(op)), _args({a0, a1}) {}
        CallNode(Atomic op, const Atomic &a0, const Atomic &a1, const Atomic &a2)
            : _op(std::move(op)), _args({a0, a1, a2}) {}
        CallNode(Atomic op, std::vector<Atomic> args) : _op(std::move(op)), _args(std::move(args)) {}
        const Atomic &op() const { return _op; }
        const std::vector<Atomic> &args() const { return _args; }
        AKR_DECL_NODE(CallNode)
        void set_op(Atomic op) { _op = std::move(op); }
        void add_arg(Atomic a) { _args.emplace_back(std::move(a)); }
    };

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

    class ConstantNode : public AtomicNode {
      public:
        AKR_DECL_NODE(ConstantNode)
        using Value = std::variant<int, float, double>;
        template <typename T, typename = std::enable_if_t<is_value<T>::value>>
        ConstantNode(const T &v) : _value(v) {
          type = (get_type_from_native<T>());
        }
        const Value &value() const { return _value; }

      private:
        Value _value;
    };

} // namespace akari::compute::ir