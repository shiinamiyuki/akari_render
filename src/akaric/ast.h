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
#include <json.hpp>
#include <akaric/base.h>
#include <akaric/lexer.h>
#include <akaric/type.h>
#include <fmt/format.h>
#include <vector>
#include <variant>
#include <unordered_set>
namespace akari::asl::ast {
    using namespace nlohmann;
    class ASTNode : public Base {
      public:
        ASTNode() = default;
        ASTNode(SourceLocation loc) : loc(loc) {}
        SourceLocation loc;
        virtual void dump_json(json &j) const {
            j = json::object();
            j["node-type"] = type_name();
            j["loc"] = fmt::format("{}:{}:{}", loc.filename, loc.line, loc.col);
        }
    };
    using AST = std::shared_ptr<ASTNode>;
    class ExpressionNode : public ASTNode {};
    class StatementNode : public ASTNode {};
    using Expr = std::shared_ptr<ExpressionNode>;

    class IdentifierNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(IdentifierNode)
        std::string identifier;
        IdentifierNode(const Token &t) {
            identifier = t.tok;
            loc = t.loc;
        }
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["var"] = identifier;
        }
    };
    using Identifier = std::shared_ptr<IdentifierNode>;

    class TypeDeclNode : public ASTNode {
      public:
        type::Qualifier qualifier = type::Qualifier::none;
    };
    using TypeDecl = std::shared_ptr<TypeDeclNode>;
    class TypenameNode : public TypeDeclNode {
      public:
        AKR_DECL_NODE(TypenameNode)
        std::string name;
        TypenameNode(const Token &t) {
            name = t.tok;
            loc = t.loc;
        }
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["typename"] = name;
        }
    };
    using Typename = std::shared_ptr<TypenameNode>;

    class LiteralNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(LiteralNode)
    };
    using Literal = std::shared_ptr<LiteralNode>;
    class IntLiteralNode : public LiteralNode {
      public:
        AKR_DECL_NODE(IntLiteralNode)
        int64_t val;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["val"] = val;
        }
    };
    using IntLiteral = std::shared_ptr<IntLiteralNode>;

    class BoolLiteralNode : public LiteralNode {
      public:
        AKR_DECL_NODE(BoolLiteralNode)
        bool val;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["val"] = val;
        }
    };
    using BoolLiteral = std::shared_ptr<BoolLiteralNode>;

    class FloatLiteralNode : public LiteralNode {
      public:
        AKR_DECL_NODE(FloatLiteralNode)
        double val;
        type::PrimitiveType type;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["val"] = val;
        }
    };
    using FloatLiteral = std::shared_ptr<FloatLiteralNode>;

    class IndexNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(IndexNode)
        Expr expr;
        Expr idx;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            expr->dump_json(j["expr"]);
            idx->dump_json(j["idx"]);
        }
    };
    using Index = std::shared_ptr<IndexNode>;
    class MemberAccessNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(MemberAccessNode)
        Expr var;
        std::string member;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            var->dump_json(j["var"]);
            j["member"] = member;
        }
    };
    using MemberAccess = std::shared_ptr<MemberAccessNode>;
    class FunctionCallNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(FunctionCallNode)
        Identifier func;
        std::vector<Expr> args;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["args"] = json::array();
            for (auto &a : args) {
                json _;
                a->dump_json(_);
                j["args"].emplace_back(_);
            }
            func->dump_json(j["func"]);
        }
    };
    using FunctionCall = std::shared_ptr<FunctionCallNode>;
    class ConstructorCallNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(ConstructorCallNode)
        Typename type;
        std::vector<Expr> args;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["args"] = json::array();
            for (auto &a : args) {
                json _;
                a->dump_json(_);
                j["args"].emplace_back(_);
            }
            type->dump_json(j["type"]);
        }
    };
    using ConstructorCall = std::shared_ptr<ConstructorCallNode>;
    class BinaryExpressionNode : public ExpressionNode {
      public:
        std::string op;
        Expr lhs, rhs;
        AKR_DECL_NODE(BinaryExpressionNode)
        BinaryExpressionNode(const Token &t) {
            op = t.tok;
            loc = t.loc;
        }
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["op"] = op;
            lhs->dump_json(j["lhs"]);
            rhs->dump_json(j["rhs"]);
        }
    };
    using BinaryExpression = std::shared_ptr<BinaryExpressionNode>;
    class ConditionalExpressionNode : public ExpressionNode {
      public:
        Expr cond;
        Expr lhs, rhs;
        AKR_DECL_NODE(ConditionalExpressionNode)
        ConditionalExpressionNode(Expr cond, Expr lhs, Expr rhs) : cond(cond), lhs(lhs), rhs(rhs) { loc = cond->loc; }
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            cond->dump_json(j["cond"]);
            lhs->dump_json(j["lhs"]);
            rhs->dump_json(j["rhs"]);
        }
    };
    using ConditionalExpression = std::shared_ptr<ConditionalExpressionNode>;
    class AssignmentNode : public StatementNode {
      public:
        std::string op;
        Expr lhs, rhs;
        AKR_DECL_NODE(AssignmentNode)
        AssignmentNode(const Token &t) {
            op = t.tok;
            loc = t.loc;
        }
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["op"] = op;
            lhs->dump_json(j["lhs"]);
            rhs->dump_json(j["rhs"]);
        }
    };
    using Assignment = std::shared_ptr<AssignmentNode>;
    class FunctionCallStatement : public StatementNode {
      public:
        FunctionCall call;
        AKR_DECL_NODE(FunctionCallStatement)
        FunctionCallStatement(FunctionCall call) : call(call) { loc = call->loc; }
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            call->dump_json(j["call"]);
        }
    };
    using CallStmt = std::shared_ptr<FunctionCallStatement>;
    class UnaryExpressionNode : public ExpressionNode {
      public:
        std::string op;
        Expr operand;
        AKR_DECL_NODE(UnaryExpressionNode)
        UnaryExpressionNode(const Token &t, Expr e) {
            op = t.tok;
            loc = t.loc;
            operand = e;
        }
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            operand->dump_json(j["operand"]);
        }
    };
     using UnaryExpression = std::shared_ptr<UnaryExpressionNode>;
    class ParameterDeclNode : public ASTNode {
      public:
        Identifier var;
        TypeDecl type;
        ParameterDeclNode(Identifier v, TypeDecl type) : var(v), type(type) { loc = v->loc; }
        AKR_DECL_NODE(ParameterDeclNode)
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            if (var)
                var->dump_json(j["var"]);
            type->dump_json(j["type"]);
        }
    };
    using ParameterDecl = std::shared_ptr<ParameterDeclNode>;
    class VarDeclNode : public ASTNode {
      public:
        Identifier var;
        TypeDecl type;
        Expr init;
        VarDeclNode(Identifier v, TypeDecl type, Expr init = nullptr) : var(v), type(type), init(init) { loc = v->loc; }
        AKR_DECL_NODE(VarDeclNode)
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            var->dump_json(j["var"]);
            type->dump_json(j["type"]);
            if (init)
                init->dump_json(j["init"]);
        }
    };
    using VarDecl = std::shared_ptr<VarDeclNode>;

    using Stmt = std::shared_ptr<StatementNode>;
    class VarDeclStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(VarDeclStatementNode)
        VarDecl decl;
        VarDeclStatementNode(VarDecl decl) : decl(decl) {}
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            decl->dump_json(j["decl"]);
        }
    };
    using VarDeclStmt = std::shared_ptr<VarDeclStatementNode>;
    class ExpressionStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(ExpressionStatementNode)
        Expr expr;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            expr->dump_json(j["expr"]);
        }
    };
    using ExpressionStatement = std::shared_ptr<ExpressionStatementNode>;
    class IfStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(IfStatementNode)
        Expr cond;
        Stmt if_true;
        Stmt if_false;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            cond->dump_json(j["cond"]);
            if_true->dump_json(j["if_true"]);
            if (if_false)
                if_false->dump_json(j["if_false"]);
        }
    };
    using IfStmt = std::shared_ptr<IfStatementNode>;
    struct Label {
        Expr value;
        bool is_default = false;
    };
    class SeqStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(SeqStatementNode)
        std::vector<Stmt> stmts;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["stmts"] = json::array();
            for (auto &st : stmts) {
                json s;
                st->dump_json(s);
                j["stmts"].push_back(s);
            }
        }
    };
    using SeqStmt = std::shared_ptr<SeqStatementNode>;
    class SwitchStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(SwitchStatementNode)
        Expr cond;
        std::vector<std::pair<std::vector<Label>, SeqStmt>> cases;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            cond->dump_json(j["cond"]);
        }
    };
    using SwitchStmt = std::shared_ptr<SwitchStatementNode>;
    class WhileStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(WhileStatementNode)
        Expr cond;
        Stmt body;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            cond->dump_json(j["cond"]);
            body->dump_json(j["body"]);
        }
    };
    using WhileStmt = std::shared_ptr<WhileStatementNode>;
    class BreakStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(BreakStatementNode)
    };
    using BreakStmt = std::shared_ptr<BreakStatementNode>;
    class ContinueStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(BreakStatementNode)
    };
    using ContinueStmt = std::shared_ptr<ContinueStatementNode>;
    class ConstDeclNode : public StatementNode {
      public:
        AKR_DECL_NODE(ConstDeclNode)
        VarDecl var;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            var->dump_json(j["var"]);
        }
    };
    using ConstDecl = std::shared_ptr<ConstDeclNode>;

    class ForStatementNode : public StatementNode {
      public:
        AKR_DECL_NODE(ForStatementNode)
        VarDecl init;
        Expr cond;
        Stmt step;
        Stmt body;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            init->dump_json(j["init"]);
            cond->dump_json(j["cond"]);
            body->dump_json(j["body"]);
            step->dump_json(j["step"]);
        }
    };
    using ForStmt = std::shared_ptr<ForStatementNode>;

    class ReturnNode : public StatementNode {
      public:
        Expr expr;
        AKR_DECL_NODE(ReturnNode)
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            expr->dump_json(j["expr"]);
        }
    };
    using Return = std::shared_ptr<ReturnNode>;
    class FunctionDeclNode : public ASTNode {
      public:
        Identifier name;
        std::vector<ParameterDecl> parameters;
        TypeDecl type;
        SeqStmt body;
        bool is_intrinsic = false;
        AKR_DECL_NODE(VarDeclNode)
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            type->dump_json(j["type"]);
            name->dump_json(j["name"]);
            if (body)
                body->dump_json(j["body"]);
        }
    };
    using FunctionDecl = std::shared_ptr<FunctionDeclNode>;
    class StructDeclNode : public TypeDeclNode {
      public:
        Typename struct_name;
        std::vector<VarDecl> fields;
        AKR_DECL_NODE(StructDeclNode)
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            struct_name->dump_json(j["name"]);
            j["fields"] = json::array();
            for (auto &s : fields) {
                auto k = json::object();
                s->dump_json(k);
                j["fields"].push_back(k);
            }
        }
    };
    using StructDecl = std::shared_ptr<StructDeclNode>;
    class ArrayDeclNode : public TypeDeclNode {
      public:
        AKR_DECL_NODE(ArrayDeclNode)
        TypeDecl element_type;
        Expr length;

        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            element_type->dump_json(j["inner"]);
            if (length)
                length->dump_json(j["length"]);
        }
    };
    using ArrayDecl = std::shared_ptr<ArrayDeclNode>;
    class BufferObjectNode : public ASTNode {
      public:
        AKR_DECL_NODE(BufferObjectNode)
        int binding = -1;
        BufferObjectNode(VarDecl var) : var(var) { loc = var->loc; }
        VarDecl var;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            var->dump_json(j["var"]);
        }
    };
    using BufferObject = std::shared_ptr<BufferObjectNode>;
    class UniformVarNode : public ASTNode {
      public:
        AKR_DECL_NODE(UniformVarNode)
        int binding = -1;
        UniformVarNode(VarDecl var) : var(var) { loc = var->loc; }
        VarDecl var;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            var->dump_json(j["var"]);
        }
    };
    using UniformVar = std::shared_ptr<UniformVarNode>;
    class ConstVarNode : public ASTNode {
      public:
        AKR_DECL_NODE(ConstVarNode)
        int binding = -1;
        ConstVarNode(VarDecl var) : var(var) {
            loc = var->loc;
            var->type->qualifier = type::Qualifier((int)var->type->qualifier | (int)type::Qualifier::const_);
        }
        VarDecl var;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            var->dump_json(j["var"]);
        }
    };
    using ConstVar = std::shared_ptr<ConstVarNode>;
    using ModuleParameter = std::variant<VarDecl, Typename>;
    class TopLevelNode;
    using TopLevel = std::shared_ptr<TopLevelNode>;
    class TopLevelNode : public ASTNode {
      public:
        AKR_DECL_NODE(TopLevelNode)
        std::vector<FunctionDecl> funcs;
        std::vector<StructDecl> structs;
        std::vector<BufferObject> buffers;
        std::vector<UniformVar> uniforms;
        std::vector<ConstVar> consts;
        std::unordered_set<std::string> typenames;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["structs"] = json::array();
            for (auto &s : structs) {
                auto k = json::object();
                s->dump_json(k);
                j["structs"].push_back(k);
            }
            j["funcs"] = json::array();
            for (auto &s : funcs) {
                auto k = json::object();
                s->dump_json(k);
                j["funcs"].push_back(k);
            }
            j["buffers"] = json::array();
            for (auto &s : buffers) {
                auto k = json::object();
                s->dump_json(k);
                j["buffers"].push_back(k);
            }
            j["uniforms"] = json::array();
            for (auto &s : uniforms) {
                auto k = json::object();
                s->dump_json(k);
                j["uniforms"].push_back(k);
            }
            j["consts"] = json::array();
            for (auto &cst : consts) {
                auto k = json::object();
                cst->dump_json(k);
                j["consts"].push_back(k);
            }
        }
    };
} // namespace akari::asl::ast