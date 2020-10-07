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
    class ModuleIdentifierNode : public ExpressionNode {
      public:
        AKR_DECL_NODE(ModuleIdentifierNode)
        ModuleIdentifierNode()=default;
        ModuleIdentifierNode(const Identifier & i):identifier(i){
          loc = i->loc;
        }
        std::vector<std::string> module_chain;
        Identifier identifier;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            identifier->dump_json(j["identifier"]);
            j["module_chain"] = json::array();
            for (auto &s : module_chain) {
                j["module_chain"].push_back(s);
            }
        }
    };
    using ModuleIdentifier = std::shared_ptr<ModuleIdentifierNode>;
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
    class ModuleTypenameNode : public TypeDeclNode {
      public:
        AKR_DECL_NODE(ModuleTypenameNode)
        ModuleTypenameNode()=default;
        ModuleTypenameNode(Typename type):name(type){
          loc = type->loc;
        }
        std::vector<std::string> module_chain;
        Typename name;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            name->dump_json(j["identifier"]);
            j["module_chain"] = json::array();
            for (auto &s : module_chain) {
                j["module_chain"].push_back(s);
            }
        }
    };
    using ModuleTypename = std::shared_ptr<ModuleTypenameNode>;
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
        ModuleIdentifier func;
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
        ModuleTypename type;
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
        std::vector<ParameterDecl> parameters;
        TypeDecl type;
        SeqStmt body;
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
            length->dump_json(j["length"]);
        }
    };
    using ArrayDecl = std::shared_ptr<ArrayDeclNode>;
    class BufferObjectNode : public ASTNode {
      public:
        AKR_DECL_NODE(BufferObjectNode)
        int binding = -1;
        std::string name;
        std::vector<VarDecl> fields;
    };
    using ModuleParameter = std::variant<VarDecl, Typename>;
    class TopLevelNode;
    using TopLevel = std::shared_ptr<TopLevelNode>;
    struct ModuleDef {
      public:
        std::string name;
        std::vector<ModuleParameter> parameters;
        std::vector<std::weak_ptr<TopLevelNode>> imported;
        inline void dump_json(json &j) const;
    };
    class TopLevelNode : public ASTNode {
      public:
        AKR_DECL_NODE(TopLevelNode)
        ModuleDef module_def;
        std::vector<FunctionDecl> funcs;
        std::vector<StructDecl> structs;
        std::unordered_set<std::string> typenames;
        void dump_json(json &j) const {
            ASTNode::dump_json(j);
            module_def.dump_json(j["module"]);
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
    inline void ModuleDef::dump_json(json &j) const {
        j["name"] = name;
        j["imported"] = json::array();
        for (auto &i : imported) {
            if (auto p = i.lock()) {
                j["imported"].push_back(p->module_def.name);
            }
        }
    }
} // namespace akari::asl::ast