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
#include <akari/asl/base.h>
#include <akari/asl/lexer.h>
#include <fmt/format.h>
namespace akari::asl::ast {
    using namespace nlohmann;
    class ASTNode : public Base {
      public:
        SourceLocation loc;
        virtual void dump_json(json &j) const {
            j = json::object();
            j["node-type"] = type_name();
            j["loc"] = fmt::format("{}:{}:{}", loc.filename, loc.line, loc.col);
        }
    };
    class ExpressionNode : public ASTNode {};
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
    class TypenameNode : public ASTNode {
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

    class IntLiteralNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(IntLiteralNode)
        int64_t val;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["val"] = val;
        }
    };
    using IntLiteral = std::shared_ptr<IntLiteralNode>;

    class FloatLiteralNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(FloatLiteralNode)
        double val;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["val"] = val;
        }
    };
    using FloatLiteral = std::shared_ptr<FloatLiteralNode>;

    class ConstructNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(FloatLiteralNode)
        std::string type;
        std::vector<Expr> args;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            j["type"] = type;
            j["args"] = json::array();
            for (auto &a : args) {
                json _;
                a->dump_json(_);
                j["args"].emplace_back(_);
            }
        }
    };
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
    using MemberAcess = std::shared_ptr<MemberAccessNode>;
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
    using BinaryExpression = std::shared_ptr<BinaryExpressionNode>;
    class VarDeclNode : public ASTNode {
      public:
        Identifier var;
        Typename type;
        Expr init;
        VarDeclNode(Identifier v, Typename type, Expr init = nullptr) : var(v), type(type), init(init) {}
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

    class StatementNode : public ASTNode {};
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
        std::vector<VarDecl> parameters;
        Typename type;
        Stmt body;
        AKR_DECL_NODE(VarDeclNode)
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            type->dump_json(j["type"]);
            name->dump_json(j["name"]);
            body->dump_json(j["body"]);
        }
    };
    using FunctionDecl = std::shared_ptr<FunctionDeclNode>;
    class StructDeclNode : public ASTNode {
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
    class TopLevelNode : public ASTNode {
      public:
        AKR_DECL_NODE(TopLevelNode)
        std::vector<FunctionDecl> funcs;
        std::vector<StructDecl> structs;
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
        }
    };
    using TopLevel = std::shared_ptr<TopLevelNode>;

} // namespace akari::asl::ast