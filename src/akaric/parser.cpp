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
#include <fstream>
#include <sstream>
#include <regex>
namespace akari::asl {
    using namespace ast;
    static std::unordered_set<std::string> qualifiers = {"const", "in", "inout", "out"};
    class Parser::Impl {
        friend class Parser;
        const TokenStream &ts;
        const std::string &filename;
        const std::string &src;
        TokenStream::const_iterator it;
        std::unordered_set<std::string> typenames;

        Parser &parser;
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
        std::string module_name;

      public:
        Impl(Parser &parser, const TokenStream &ts, const std::string &filename, const std::string &src)
            : parser(parser), ts(ts), filename(filename), src(src) {
            it = ts.cbegin();
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
            typenames.insert("void");
            typenames.insert("bool");
            typenames.insert("int");
            typenames.insert("uint");
            typenames.insert("float");
            typenames.insert("double");
            for (int i = 1; i <= 4; i++) {
                typenames.insert(fmt::format("ivec{}", i));
                typenames.insert(fmt::format("bvec{}", i));
                typenames.insert(fmt::format("vec{}", i));
                typenames.insert(fmt::format("uvec{}", i));
                typenames.insert(fmt::format("dvec{}", i));
            }
            for (int m = 2; m <= 4; m++) {
                for (int n = 2; n <= 4; n++) {
                    typenames.insert(fmt::format("mat{}x{}", n, m));
                    typenames.insert(fmt::format("dmat{}x{}", n, m));
                }
                typenames.insert(fmt::format("mat{}", m));
                typenames.insert(fmt::format("dmat{}", m));
            }
        }
        [[noreturn]] void error(const SourceLocation &loc, std::string &&msg) {
            throw std::runtime_error(fmt::format("error: {} at {}:{}:{}", msg, loc.filename, loc.line, loc.col));
        }
        ast::ConstVar parse_const_decl() {
            expect("const");
            auto p = std::make_shared<ConstVarNode>(parse_var_decl_must_init());
            expect(";");
            return p;
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
            if (cur().tok == "?") {
                consume();
                auto cond = result;
                auto lhs = parse_expr();
                expect(":");
                auto rhs = parse_expr();
                return std::make_shared<ConditionalExpressionNode>(cond, lhs, rhs);
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
            if (cur().tok == "~" || cur().tok == "!" || cur().tok == "-" || cur().tok == "*") {
                return parse_unary();
            }
            if (typenames.find(cur().tok) != typenames.end()) {
                auto ty = parse_typename();
                if (cur().tok == ".") {
                    // typename.method is a special kind of function name
                    // translates to typename_method
                    consume();
                    auto id = parse_identifier();
                    id->identifier = ty->name + "_" + id->identifier;
                    auto call = std::make_shared<FunctionCallNode>();
                    call->func = id;
                    call->args = parse_args();
                    call->loc = call->func->loc;
                    return call;
                }
                auto call = std::make_shared<ConstructorCallNode>();
                call->type = ty;
                call->args = parse_args();
                return call;
            }
            auto e = parse_atom();
            while (!end() && (cur().tok == "." || cur().tok == "[")) {
                if (cur().tok == ".") {
                    auto loc = cur().loc;

                    consume();
                    if (cur().type == TokenType::identifier) {
                        auto access = std::make_shared<MemberAccessNode>();
                        access->loc = loc;
                        auto m = parse_identifier();
                        access->var = e;
                        access->member = m->identifier;
                        e = access;
                    } else {
                        error(cur().loc, "unexpected token in member access expression");
                    }
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
                    call->loc = call->func->loc;
                    return call;
                }
                error(cur().loc, "identifier expected in function call");
            }
            return e;
        }
        ast::TupleExpression parse_tuple() {
            expect("(");
            std::vector<ast::Expr> elements;
            elements.push_back(parse_expr());
            expect(",");
            while (cur().tok != ")") {
                elements.push_back(parse_expr());
                if (cur().tok != ")")
                    expect(",");
            }
            expect(")");
            return std::make_shared<TupleExpressionNode>(elements);
        }
        ast::UnaryExpression parse_unary() {
            if (cur().tok == "-" || cur().tok == "!" || cur().tok == "~") {
                auto op = cur();
                consume();
                auto e = parse_postfix_expr();
                return std::make_shared<ast::UnaryExpressionNode>(op, e);
            } else {
                error(cur().loc, "illegal token in unary expression");
            }
        }
        ast::Expr parse_atom() {
            if (cur().type == identifier) {
                auto iden = parse_identifier();
                return iden;
            }
            if (cur().tok == "(") {
                consume();
                ast::Expr e = parse_expr();
                if (cur().tok == ",") {
                    std::vector<ast::Expr> elements;
                    elements.push_back(e);
                    consume();
                    if (cur().tok != ")") {
                        elements.push_back(parse_expr());
                        while (cur().tok == ",") {
                            consume();
                            elements.push_back(parse_expr());
                        }
                    }
                    expect(")");
                    return std::make_shared<TupleExpressionNode>(elements);
                }
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
            if (cur().tok == "true") {
                auto lit = std::make_shared<ast::BoolLiteralNode>();
                lit->val = true;
                lit->loc = cur().loc;
                consume();
                return lit;
            }
            if (cur().tok == "false") {
                auto lit = std::make_shared<ast::BoolLiteralNode>();
                lit->val = false;
                lit->loc = cur().loc;
                consume();
                return lit;
            }
            error(cur().loc, fmt::format("unexpected token {}", cur().tok));
            return nullptr;
        }

        ast::TypeDecl parse_typedecl() {
            ast::TypeDecl p = parse_typename();
            if (cur().tok == "[") {
                std::vector<Expr> lengths;
                while (cur().tok == "[") {
                    consume();
                    ast::Expr length = cur().tok == "]" ? nullptr : parse_expr();
                    expect("]");
                    lengths.emplace_back(length);
                }
                std::reverse(lengths.begin(), lengths.end());
                for (auto length : lengths) {
                    auto arr = std::make_shared<ast::ArrayDeclNode>();
                    arr->loc = p->loc;
                    arr->element_type = p;
                    arr->length = length;
                    p = arr;
                }
            }
            return p;
        }
        bool is_qualifier() { return qualifiers.find(cur().tok) != qualifiers.end(); }
        ast::Typename parse_typename() {
            if (typenames.find(cur().tok) != typenames.end() || is_qualifier()) {
                type::Qualifier qualifier = type::Qualifier::none;
                if (cur().tok == "in") {
                    consume();
                    qualifier = type::Qualifier::none;
                } else if (cur().tok == "out") {
                    consume();
                    qualifier = type::Qualifier::out;
                } else if (cur().tok == "inout") {
                    consume();
                    qualifier = type::Qualifier::inout;
                } else if (cur().tok == "const") {
                    consume();
                    qualifier = type::Qualifier::constant;
                }
                if (typenames.find(cur().tok) == typenames.end()) {
                    error(cur().loc, "typename expected");
                }
                auto p = std::make_shared<ast::TypenameNode>(cur());
                p->qualifer = qualifier;
                consume();
                return p;
            } else {
                error(cur().loc, fmt::format("unknown typename {}", cur().tok));
            }
        }
        // std::pair<ast::ModuleIdentifier, ast::ModuleTypename> parse_module_function_or_typename() {}
        ast::TypeDecl parse_struct_decl() {
            expect("struct");
            std::unordered_set<std::string> fields;
            if (cur().type == identifier) {
                if (typenames.find(cur().tok) != typenames.end()) {
                    error(cur().loc, fmt::format("{} redefined", cur().tok));
                }
                typenames.insert(cur().tok);
            } else {
                error(cur().loc, "identifier expected");
            }
            auto name = parse_typename();
            auto st = std::make_shared<StructDeclNode>();
            st->struct_name = name;
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
            auto ty = parse_typedecl();
            auto iden = parse_identifier();
            return std::make_shared<VarDeclNode>(iden, ty, nullptr);
        }
        ast::VarDecl parse_var_decl_init() {
            auto ty = parse_typedecl();
            auto iden = parse_identifier();
            if (cur().tok == "=") {
                consume();
                auto init = parse_expr();
                return std::make_shared<VarDeclNode>(iden, ty, init);
            }
            return std::make_shared<VarDeclNode>(iden, ty, nullptr);
        }
        ast::VarDecl parse_var_decl_must_init() {
            auto ty = parse_typedecl();
            auto iden = parse_identifier();

            expect("=");
            auto init = parse_expr();
            return std::make_shared<VarDeclNode>(iden, ty, init);

            return std::make_shared<VarDeclNode>(iden, ty, nullptr);
        }
        ast::ParameterDecl parse_parameter_decl() {
            auto ty = parse_typedecl();
            auto iden = parse_identifier();
            return std::make_shared<ParameterDeclNode>(iden, ty);
        }
        ast::VarDeclStmt parse_var_decl_stmt() {
            auto st = std::make_shared<VarDeclStatementNode>(parse_var_decl_init());
            expect(";");
            return st;
        }
        Label parse_label() {
            if (cur().tok != "case" && cur().tok != "default") {
                error(cur().loc, fmt::format("expect 'case' or 'default' but found {}", cur().tok));
            }
            Label label;
            if (cur().tok == "default") {
                consume();
                label = Label{nullptr, true};
            } else {
                consume();
                label = Label{parse_expr(), false};
            }
            expect(":");
            return label;
        }
        ast::SwitchStmt parse_switch() {
            expect("switch");
            auto w = std::make_shared<SwitchStatementNode>();
            expect("(");
            w->cond = parse_expr();
            expect(")");
            expect("{");
            while (cur().tok != "}") {
                auto labels = std::vector<Label>();
                labels.emplace_back(parse_label());
                while (cur().tok == "case" || cur().tok == "default") {
                    labels.emplace_back(parse_label());
                }
                auto body = parse_block();
                w->cases.emplace_back(std::make_pair(std::move(labels), body));
            }
            expect("}");
            return w;
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
            w->init = parse_var_decl_init();
            expect(";");
            w->cond = parse_expr();
            expect(";");
            w->step = parse_expr_stmt();
            expect(")");
            w->body = parse_stmt();
            return w;
            return nullptr;
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

        ast::Stmt parse_expr_stmt() {
            auto lvalue = parse_postfix_expr();
            if (!assignOps.count(cur().tok)) {
                // function call
                if (!lvalue->isa<ast::FunctionCall>()) {
                    error(cur().loc, fmt::format("an expression statment but be assignment or function call"));
                }
                return std::make_shared<ast::FunctionCallStatement>(lvalue->cast<ast::FunctionCall>());
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
            if (is_qualifier() || typenames.find(cur().tok) != typenames.end()) {
                return parse_var_decl_stmt();
            }
            if (cur().tok == "while") {
                return parse_while();
            }
            if (cur().tok == "switch") {
                return parse_switch();
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
            auto st = parse_expr_stmt();
            expect(";");
            return st;
        }
        ast::FunctionDecl parse_intrinsic() {
            expect("__builtin__");
            auto func = parse_func_decl();
            func->is_intrinsic = true;
            return func;
        }
        ast::FunctionDecl parse_func_decl() {
            auto func = std::make_shared<FunctionDeclNode>();
            func->type = parse_typedecl();
            func->name = parse_identifier();
            expect("(");
            while (!end() && cur().tok != ")") {
                func->parameters.emplace_back(parse_parameter_decl());
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

        ast::TopLevel parse() {
            module_name = fs::path(filename).stem().string();
            auto top = std::make_shared<TopLevelNode>();
            while (!end()) {
                if (cur().tok == "struct") {
                    top->structs.emplace_back(parse_struct_decl());
                    top->defs.emplace_back(top->structs.back());
                } else if (cur().tok == "const") {
                    top->consts.emplace_back(parse_const_decl());
                    top->defs.emplace_back(top->consts.back());
                } else if (cur().tok == "__builtin__") {
                    top->funcs.emplace_back(parse_intrinsic());
                    top->defs.emplace_back(top->funcs.back());
                } else if (is_qualifier() || typenames.find(cur().tok) != typenames.end()) {
                    top->funcs.emplace_back(parse_func_decl());
                    top->defs.emplace_back(top->funcs.back());
                } else if (cur().type == TokenType::pp_include) {
                    parser.handle_include(cur().tok, this);
                    consume();
                } else {
                    error(cur().loc, fmt::format("unexpected token {}", cur().tok));
                }
            }
            top->typenames = typenames;
            return top;
        }
    };
    void Parser::handle_include(const std::string &path, Impl *impl) {
        auto full_path = fs::absolute(fs::path(path)).string();
        if (header_types.find(full_path) != header_types.end()) {
            for (auto &ty : header_types.at(full_path)) {
                impl->typenames.insert(ty);
            }
        } else {
            header_types[full_path] = {};
            auto tree = parse(full_path);
            tree->is_header = true;
            header_types[full_path] = tree->typenames;
            headers.emplace_back(TranslationUnit{full_path, tree});
        }
    }
    Parser::Parser() {}
    ast::TopLevel Parser::parse(const std::string &full_path) {
        std::ifstream in(full_path);
        std::stringstream buf;
        buf << in.rdbuf();
        auto src = buf.str();
        auto ts = Lexer()(full_path, src);
        Impl impl(*this, ts, full_path, src);
        for (auto &t : type_parameters) {
            impl.typenames.insert(t);
        }
        auto tree = impl.parse();
        tree->typenames = impl.typenames;
        return tree;
    }

    std::vector<TranslationUnit> Parser::operator()(const std::vector<std::string> &filenames) {
        std::vector<TranslationUnit> units;
        for (auto &filename : filenames) {
            TranslationUnit unit;
            auto full_path = fs::absolute(fs::path(filename)).string();
            unit.tree = parse(full_path);
            unit.filename = full_path;
            units.emplace_back(unit);
        }
        units.insert(units.begin(), headers.begin(), headers.end());
        return units;
    }
} // namespace akari::asl