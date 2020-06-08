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
#include <akari/compute/types.hpp>

namespace akari::compute::ir {
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
    class Node : public Base {
      public:
        virtual void accept(NodeVisitor &) = 0;
    };
    using NodePtr = std::shared_ptr<Node>;
    using VarNodePtr = std::shared_ptr<VarNode>;
    using FunctionNodePtr = std::shared_ptr<FunctionNode>;
#define AKR_DECL_NODE(Type)                                                                                            \
    std::string type_name() const { return #Type; }                                                                    \
    void accept(NodeVisitor &vis) { vis.visit(*this); }
    // class Expr : public Node {};
    using Expr = Node;
    class VarNode : public Expr {
        size_t id;
        std::string _name;

      public:
        explicit VarNode(size_t id):id(id), _name(std::to_string(id)){}
        VarNode(size_t id, std::string name) : id(id), _name(std::move(name)) {}
        auto & name()const{return _name;}
        AKR_DECL_NODE(VarNode)
    };

    enum class Primitive {
        EAdd,
        ESub,
        EMul,
        EDiv,
        EAnd,
        EOr,
        EXor,
    };

    class PrimitiveNode : public Expr {
        Primitive _prim;

      public:
        PrimitiveNode(Primitive p) : _prim(p) {}
        Primitive primitive() const { return _prim; }
        AKR_DECL_NODE(PrimitiveNode)
    };

    // Anonymous Function Node
    class FunctionNode : public Expr {
        std::vector<std::shared_ptr<VarNode>> _parameters;
        std::shared_ptr<Expr> _body;

      public:
        FunctionNode(std::vector<std::shared_ptr<VarNode>> parameters, std::shared_ptr<Expr> body)
            : _parameters(std::move(parameters)), _body(std::move(body)) {}
        const std::vector<std::shared_ptr<VarNode>> &parameters() const { return _parameters; }
        auto body() const { return _body; }
        AKR_DECL_NODE(FunctionNode)
    };
    class IfNode : public Expr {
        std::shared_ptr<Node> _cond, _then, _else;

      public:
        IfNode(std::shared_ptr<Node> _cond, std::shared_ptr<Node> _then, std::shared_ptr<Node> _else)
            : _cond(std::move(_cond)), _then(std::move(_then)), _else(std::move(_else)) {}
        AKR_DECL_NODE(IfNode)
        const std::shared_ptr<Node> &cond() const { return _cond; }
        const std::shared_ptr<Node> &then() const { return _then; }
        const std::shared_ptr<Node> &else_() const { return _else; }
    };
    class LetNode : public Expr {
        VarNodePtr _var;
        std::shared_ptr<Node>  _val, _body;

      public:
      public:
        LetNode(VarNodePtr _var, std::shared_ptr<Node> _val, std::shared_ptr<Node> _body)
            : _var(std::move(_var)), _val(std::move(_val)), _body(std::move(_body)) {}
        const VarNodePtr &var() const { return _var; }
        const std::shared_ptr<Node> &value() const { return _val; }
        const std::shared_ptr<Node> &body() const { return _body; }
        AKR_DECL_NODE(LetNode)
    };
    class CallNode : public Expr {
        std::shared_ptr<Node> _op;
        std::vector<std::shared_ptr<Expr>> _args;

      public:
        CallNode()=default;
        CallNode(std::shared_ptr<Node> op, std::vector<std::shared_ptr<Expr>> args)
            : _op(std::move(op)), _args(std::move(args)) {}
        const std::shared_ptr<Node> &op() const { return _op; }
        const std::vector<std::shared_ptr<Expr>> &args() const { return _args; }
        AKR_DECL_NODE(CallNode)
        void set_op(std::shared_ptr<Node> op) { _op = std::move(op); }
        void add_arg(std::shared_ptr<Expr> a) { _args.emplace_back(std::move(a)); }
    };

    template <typename T> struct is_value : std::false_type {};
    template <> struct is_value<int> : std::true_type {};
    template <> struct is_value<float> : std::true_type {};

    template <typename T> std::shared_ptr<Expr> to_expr(T &&v) {
        if constexpr (is_value<T>::value) {
            return std::make_shared<ConstantNode>(v);
        } else {
            return v;
        }
    }
    template <typename... Ts> std::shared_ptr<CallNode> call(Primitive op, Ts &&... _args) {
        std::vector<std::shared_ptr<Expr>> args = {to_expr(std::forward<Ts>(_args))...};
        auto cnode = std::make_shared<CallNode>();
        cnode->set_op(std::make_shared<PrimitiveNode>(op));
        for (auto &a : args) {
            cnode->add_arg(a);
        }
        return cnode;
    }
    class ConstantNode : public Expr {
      public:
        AKR_DECL_NODE(ConstantNode)
        using Value = std::variant<int, float, double>;
        template <typename T, typename = std::enable_if_t<is_value<T>::value>> ConstantNode(const T &v) : _value(v) {}
        const Value &value() const { return _value; }

      private:
        Value _value;
    };
} // namespace akari::compute::ir