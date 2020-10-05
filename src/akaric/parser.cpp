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
#include <optional>
#include <akaric/parser.h>
#include <iostream>
#include <fmt/format.h>
#include <unordered_set>
namespace akari::asl {
    using namespace ast;
    class Parser::Impl {
        friend class Parser;
        TokenStream ts;
        TokenStream::iterator it;
        Environment<std::string, bool> typenames;
        const Token &cur() {
            if (it == ts.end()) {
                throw std::runtime_error("unexpected EOF");
            }
            return *it;
        }
        bool end() { return it == ts.end(); }
        void consume() {
            if (it != ts.end()) {
                // std::cout << it->tok << std::endl;
                it++;
            }
        }
        std::optional<Token> peek(int i = 1) {
            auto q = it;
            for (int _ = 0; _ < i; _++) {
                if (q != ts.end())
                    q++;
            }
            if (q == ts.end()) {
                return std::nullopt;
            }
            return *q;
        }
        OperatorPrecedence prec;
        std::unordered_set<std::string> assignOps;
        void expect(const std::string &s) {
            if (cur().tok != s) {
                error(cur().loc, fmt::format("expect '{}' but found '{}'", s, cur().tok));
            }
            consume();
        }

      public:
        Impl() {

            // for(auto & t: ts){
            //     std::cout << t.tok << std::endl;
            // }
            /*
             *   opPrec[","] = prec;
             * prec++;*/
            // opPrec["="] = prec;
            // opPrec["+="] = prec;
            // opPrec["-="] = prec;
            // opPrec["*="] = prec;
            // opPrec["/="] = prec;
            // opPrec[">>="] = prec;
            // opPrec["<<="] = prec;
            // opPrec["%="] = prec;
            // opPrec["|="] = prec;
            // opPrec["&="] = prec;
            // opPrec["^="] = prec;
            // opPrec["&&="] = prec;
            // opPrec["||="] = prec;
            // prec++;

            assignOps = {
                "=", "+=", "-=", "*=", "/=", ">>=", "<<=", "%=", "|=", "&=", "^=", "&&=", "||=",
            };
            typenames.insert("bool", true);
            typenames.insert("int", true);
            typenames.insert("uint", true);
            typenames.insert("float", true);
            typenames.insert("double", true);
            for (int i = 1; i <= 4; i++) {
                typenames.insert(fmt::format("ivec{}", i), true);
                typenames.insert(fmt::format("bvec{}", i), true);
                typenames.insert(fmt::format("vec{}", i), true);
                typenames.insert(fmt::format("dvec{}", i), true);
            }
            for (int m = 2; m <= 4; m++) {
                for (int n = 2; n <= 4; n++) {
                    typenames.insert(fmt::format("imat{}{}", n, m), true);
                    typenames.insert(fmt::format("bmat{}{}", n, m), true);
                    typenames.insert(fmt::format("mat{}{}", n, m), true);
                    typenames.insert(fmt::format("dmat{}{}", n, m), true);
                }
            }
        }
        [[noreturn]] void error(const SourceLocation &loc, std::string &&msg) {
            throw std::runtime_error(fmt::format("error: {} at {}:{}:{}", msg, loc.filename, loc.line, loc.col));
        }
        ast::Expr parse_expr(int lev = 0) {
            ast::Expr result = parse_postfix_expr();
            while (!end()) {
                auto c = cur();
                // std::cout << c.tok << std::endl;
                if (prec.opPrec.find(c.tok) == prec.opPrec.end())
                    break;
                if (prec.opPrec[c.tok] >= lev) {
                    consume();
                    ast::Expr rhs = parse_expr(prec.opAssoc[c.tok] + prec.opPrec[c.tok]);
                    ast::BinaryExpression op = std::make_shared<ast::BinaryExpressionNode>(c);
                    op->lhs = result;
                    op->rhs = rhs;
                    result = op;
                } else
                    break;
            }
            return result;
        }
        ast::Identifier parse_identifier() {
            if (cur().type == identifier) {
                auto p = std::make_shared<ast::IdentifierNode>(cur());
                consume();
                return p;
            }
            error(cur().loc, fmt::format("identifier expected but found {}", cur().tok));
        }
        std::vector<Expr> parse_args() {
            expect("(");
            std::vector<Expr> args;
            while (!end() && cur().tok != ")") {
                args.emplace_back(parse_expr());
                if (cur().tok == ",") {
                    consume();
                    if (cur().tok == ")") {
                        error(cur().loc, "extra ',' ");
                    }
                } else {
                    break;
                }
            }
            expect(")");
            return args;
        }
        ast::Expr parse_postfix_expr() {
            if (typenames.at(cur().tok).has_value()) {
                auto ty = parse_typename();
                auto call = std::make_shared<ConstructorCallNode>();
                call->type = ty;
                call->args = parse_args();
                return call;
            }
            auto e = parse_atom();
            while (!end() && (cur().tok == "." || cur().tok == "[")) {
                if (cur().tok == ".") {
                    auto access = std::make_shared<MemberAccessNode>();
                    access->loc = cur().loc;
                    consume();
                    auto m = parse_identifier();
                    access->var = e;
                    access->member = m->identifier;
                    e = access;
                }
                if (cur().tok == "[") {
                    consume();
                    auto i = parse_expr();
                    expect("]");
                    auto idx = std::make_shared<IndexNode>();
                    idx->expr = e;
                    idx->idx = i;
                    e = idx;
                }
            }
            if (cur().tok == "(") {
                if (e->isa<Identifier>()) {
                    auto call = std::make_shared<FunctionCallNode>();
                    call->func = e->cast<Identifier>();
                    call->args = parse_args();
                    return call;
                }
                error(cur().loc, "identifier expected in function call");
            }
            return e;
        }
        ast::Expr parse_atom() {
            if (cur().type == identifier) {
                auto iden = parse_identifier();

                return iden;
            }
            if (cur().tok == "-" || cur().tok == "!" || cur().tok == "~") {
                auto op = cur();
                consume();
                auto e = parse_atom();
                return std::make_shared<ast::UnaryExpressionNode>(op, e);
            }
            if (cur().tok == "(") {
                consume();
                auto e = parse_expr();
                expect(")");
                return e;
            }
            if (cur().type == int_literal) {
                auto lit = std::make_shared<ast::IntLiteralNode>();
                lit->val = std::stoll(cur().tok);
                lit->loc = cur().loc;
                consume();
                return lit;
            }
            if (cur().type == float_literal) {
                auto lit = std::make_shared<ast::FloatLiteralNode>();
                lit->val = std::stod(cur().tok);
                lit->loc = cur().loc;
                consume();
                return lit;
            }
            error(cur().loc, fmt::format("unexpected token {}", cur().tok));
            return nullptr;
        }
        ast::Typename parse_typename() {
            if (typenames.at(cur().tok).has_value()) {
                auto p = std::make_shared<ast::TypenameNode>(cur());
                consume();
                return p;
            }
            AKR_ASSERT(false);
        }
        ast::StructDecl parse_struct_decl() {
            expect("struct");
            std::unordered_set<std::string> fields;
            if (cur().type == identifier) {
                if (typenames.at(cur().tok).has_value()) {
                    error(cur().loc, fmt::format("{} redefined", cur().tok));
                }
                typenames.insert(cur().tok, true);
            } else {
                error(cur().loc, "identifier expected");
            }
            auto st = std::make_shared<StructDeclNode>();
            st->struct_name = parse_typename();
            expect("{");
            while (!end() && cur().tok != "}") {
                auto decl = parse_var_decl();
                st->fields.emplace_back(decl);
                if (fields.count(decl->var->identifier)) {
                    error(decl->loc, fmt::format("field {} alreay exists", decl->var->identifier));
                }
                fields.insert(decl->var->identifier);
                expect(";");
            }
            expect("}");
            expect(";");
            return st;
        }
        ast::VarDecl parse_var_decl() {
            auto ty = parse_typename();
            auto iden = parse_identifier();
            if (cur().tok == "=") {
                consume();
                auto init = parse_expr();
                return std::make_shared<VarDeclNode>(iden, ty, init);
            }
            return std::make_shared<VarDeclNode>(iden, ty, nullptr);
        }
        ast::VarDeclStmt parse_var_decl_stmt() {
            auto st = std::make_shared<VarDeclStatementNode>(parse_var_decl());
            expect(";");
            return st;
        }
        ast::WhileStmt parse_while() {
            expect("while");
            auto w = std::make_shared<WhileStatementNode>();
            expect("(");
            w->cond = parse_expr();
            expect(")");
            w->body = parse_stmt();
            return w;
        }
        ast::ForStmt parse_for() {
            expect("for");
            auto w = std::make_shared<ForStatementNode>();
            expect("(");
            w->init = parse_var_decl();
            expect(";");
            w->cond = parse_expr();
            expect(";");
            w->step = parse_assignment();
            expect(")");
            w->body = parse_stmt();
            return w;
        }
        ast::SeqStmt parse_block() {
            expect("{");
            auto seq = std::make_shared<SeqStatementNode>();
            while (!end() && cur().tok != "}") {
                seq->stmts.emplace_back(parse_stmt());
            }
            expect("}");
            return seq;
        }
        ast::IfStmt parse_if() {
            expect("if");
            auto i = std::make_shared<IfStatementNode>();
            expect("(");
            i->cond = parse_expr();
            expect(")");
            i->if_true = parse_stmt();
            if (cur().tok == "else") {
                consume();
                i->if_false = parse_stmt();
            }
            return i;
        }
        ast::Return parse_ret() {
            expect("return");
            auto ret = std::make_shared<ReturnNode>();
            ret->expr = parse_expr();
            expect(";");
            return ret;
        }
        ast::Assignment parse_assignment() {
            auto lvalue = parse_postfix_expr();
            if (!assignOps.count(cur().tok)) {
                error(cur().loc, "assignment op expected");
            }
            auto op = cur();
            consume();
            auto e = parse_expr();
            auto st = std::make_shared<ast::AssignmentNode>(op);
            st->lhs = lvalue;
            st->rhs = e;
            return st;
        }
        ast::Stmt parse_stmt() {
            if (typenames.at(cur().tok).has_value()) {
                return parse_var_decl_stmt();
            }
            if (cur().tok == "while") {
                return parse_while();
            }
            if (cur().tok == "for") {
                return parse_for();
            }
            if (cur().tok == "if") {
                return parse_if();
            }
            if (cur().tok == "{") {
                return parse_block();
            }
            if (cur().tok == "return") {
                return parse_ret();
            }
            if (cur().tok == "break") {
                consume();
                expect(";");
                return std::make_shared<BreakStatementNode>();
            }
            if (cur().tok == "continue") {
                consume();
                expect(";");
                return std::make_shared<ContinueStatementNode>();
            }
            auto st = parse_assignment();
            expect(";");
            return st;
        }
        ast::FunctionDecl parse_func_decl() {
            auto func = std::make_shared<FunctionDeclNode>();
            func->type = parse_typename();
            func->name = parse_identifier();
            expect("(");
            while (!end() && cur().tok != ")") {
                func->parameters.emplace_back(parse_var_decl());
                if (cur().tok == ",") {
                    consume();
                    if (cur().tok == ")") {
                        error(cur().loc, "extra ',' ");
                    }
                } else {
                    break;
                }
            }
            expect(")");
            if (cur().tok == "{") {
                func->body = parse_block();
            } else {
                expect(";");
            }
            return func;
        }
        ast::TopLevel operator()(const std::string &filename, const std::string &src) {
            ts = Lexer()(filename, src);
            it = ts.begin();

            auto top = std::make_shared<TopLevelNode>();
            while (!end()) {
                if (cur().tok == "struct") {
                    top->structs.emplace_back(parse_struct_decl());
                }
                if (typenames.at(cur().tok).has_value()) {
                    top->funcs.emplace_back(parse_func_decl());
                } else {
                    error(cur().loc, fmt::format("unknown type name {}", cur().tok));
                }
            }
            return top;
        }
    };
    Parser::Parser() { impl = std::make_shared<Impl>(); }
    void Parser::add_type_parameter(const std::string &type) { impl->typenames.insert(type, true); }
    ast::TopLevel Parser::operator()(const std::string &filename, const std::string &src) {
        return (*impl)(filename, src);
    }
} // namespace akari::asl