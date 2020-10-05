
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

#include "codegen.h"
#include <sstream>
namespace akari::asl {
    CodeGenerator::CodeGenerator() { add_predefined_types(); }
    static type::Type create_vec_type(const type::Type &base, int n) {
        auto v = std::make_shared<type::VectorTypeNode>();
        v->element_type = base;
        v->count = n;
        return v;
    }
    void CodeGenerator::add_type_parameters() {
        for (auto &p : module.type_parameters) {
            types[p] = std::make_shared<type::OpaqueTypeNode>(p);
        }
    }
    void CodeGenerator::add_predefined_types() {
        types["bool"] = type::boolean;
        types["int"] = type::int32;
        types["uint"] = type::uint32;
        types["double"] = type::float64;
        types["float"] = type::float32;
        for (int i = 2; i <= 4; i++) {
            types[fmt::format("vec{}", i)] = create_vec_type(type::float32, i);
            types[fmt::format("ivec{}", i)] = create_vec_type(type::int32, i);
            types[fmt::format("bvec{}", i)] = create_vec_type(type::boolean, i);
            types[fmt::format("dvec{}", i)] = create_vec_type(type::float64, i);
        }
    }
    type::Type CodeGenerator::process_type(const ast::AST &n) {
        if (n->isa<ast::VarDecl>()) {
            return process_type(n->cast<ast::VarDecl>()->type);
        }
        if (n->isa<ast::Typename>()) {
            auto ty = n->cast<ast::Typename>();
            if (!types.count(ty->name)) {
                throw std::runtime_error(fmt::format("definition of type {} not found", ty->name));
            }
            return types.at(ty->name);
        }
        if (n->isa<ast::StructDecl>()) {
            return process_struct_decl(n->cast<ast::StructDecl>());
        }
        if (n->isa<ast::FunctionDecl>()) {
            auto func = n->cast<ast::FunctionDecl>();
            auto ty = std::make_shared<type::FunctionTypeNode>();
            for (auto &a : func->parameters) {
                ty->args.emplace_back(process_type(a->type));
            }
            ty->ret = process_type(func->type);
            return ty;
        }
        AKR_ASSERT(false);
    }
    type::StructType CodeGenerator::process_struct_decl(const ast::StructDecl &decl) {
        auto st = std::make_shared<type::StructTypeNode>();
        for (auto &field : decl->fields) {
            auto t = process_type(field->type);
            type::StructField f;
            f.index = (int)st->fields.size();
            f.name = field->var->identifier;
            f.type = t;
            st->fields.emplace_back(f);
            return st;
        }
        st->name = decl->struct_name->name;
        types.emplace(st->name, st);
        structs[st->name] = st;
        return st;
    }
    void CodeGenerator::process_struct_decls() {
        for (auto &m : module.translation_units) {
            for (auto &decl : m->structs) {
                (void)process_struct_decl(decl);
            }
        }
    }
    void CodeGenerator::process_prototypes() {
        for (auto &m : module.translation_units) {
            for (auto &decl : m->funcs) {
                auto f_ty = process_type(decl)->cast<type::FunctionType>();
                prototypes[decl->name->identifier].overloads.emplace(
                    Mangler().mangle(decl->name->identifier, f_ty->args), f_ty);
            }
        }
    }
    class CodeGenCPP : public CodeGenerator {
        bool loop_pred = false;
        bool is_cuda;
        OperatorPrecedence prec;

      public:
        CodeGenCPP(bool is_cuda) : is_cuda(is_cuda) {}
        static std::string type_to_str(const type::Type &ty) {
            if (ty == type::float32) {
                return "Float";
            }
            if (ty == type::float64) {
                return "double";
            }
            if (ty == type::int32) {
                return "int";
            }
            if (ty == type::uint32) {
                return "uint";
            }
            if (ty == type::boolean) {
                return "bool";
            }
            if (ty->isa<type::OpaqueType>()) {
                return ty->cast<type::OpaqueType>()->name;
            }
            if (ty->isa<type::VectorType>()) {
                auto v = ty->cast<type::VectorType>();
                auto et = v->element_type;
                auto n = v->count;
                if (ty == type::uint32) {
                    return fmt::format("uint{}", n);
                } else {
                    return fmt::format("{}{}", type_to_str(et), n);
                }
            }
            if (ty->isa<type::StructType>()) {
                return ty->cast<type::StructType>()->name;
            }
            AKR_ASSERT(false);
        }
        ValueRecord compile_var(const ast::Identifier &var) {
            if (!vars.at(var->identifier).has_value()) {
                error(var->loc, fmt::format("identifier {} not found", var->identifier));
            }
            auto r = vars.at(var->identifier).value();
            return {var->identifier, r.type};
        }
        ValueRecord compile_literal(const ast::Literal &lit) {
            if (lit->isa<ast::FloatLiteral>()) {
                auto fl = lit->cast<ast::FloatLiteral>();
                return {fmt::format("{}", fl->val), type::float32};
            }
            if (lit->isa<ast::IntLiteral>()) {
                auto i = lit->cast<ast::IntLiteral>();
                return {fmt::format("{}", i->val), type::int32};
            }
            AKR_ASSERT(false);
        }
        std::tuple<type::Type, ValueRecord, ValueRecord> check_binary_expr(const std::string &op,
                                                                           const SourceLocation &loc,
                                                                           const ValueRecord &lhs,
                                                                           const ValueRecord &rhs) {
            if (op == "||" || op == "&&") {
                if (lhs.type != type::boolean || rhs.type != type::boolean) {
                    error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type->type_name(),
                                           rhs.type->type_name()));
                }
                return std::make_tuple(type::boolean, lhs, rhs);
            }

            if (op == "+" || op == "-" || op == "*" || op == "/") {
                if (lhs.type == rhs.type) {
                    return std::make_tuple(lhs.type, lhs, rhs);
                }
                if (lhs.type->isa<type::VectorType>() && rhs.type->isa<type::PrimitiveType>()) {
                    auto vt = lhs.type->cast<type::VectorType>();
                    auto et = vt->element_type;
                    if (et != rhs.type) {
                        error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type->type_name(),
                                               rhs.type->type_name()));
                    }
                    return std::make_tuple(lhs.type, lhs, ValueRecord{rhs.value, lhs.type});
                } else if (rhs.type->isa<type::VectorType>() && lhs.type->isa<type::PrimitiveType>()) {
                    auto vt = rhs.type->cast<type::VectorType>();
                    auto et = vt->element_type;
                    if (et != rhs.type) {
                        error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type->type_name(),
                                               rhs.type->type_name()));
                    }
                    return std::make_tuple(rhs.type, ValueRecord{lhs.value, rhs.type}, rhs);
                } else {
                    error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type->type_name(),
                                           rhs.type->type_name()));
                }
            } else if (op == "<" || op == "<=" || op == ">" || op == ">=" || op == "!=" || op == "==") {
                if (lhs.type->isa<type::VectorType>() || rhs.type->isa<type::VectorType>()) {
                    error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type->type_name(),
                                           rhs.type->type_name()));
                }
                if (lhs.type != rhs.type) {
                    error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type->type_name(),
                                           rhs.type->type_name()));
                }
                return std::make_tuple(type::boolean, lhs, rhs);
            }

            AKR_ASSERT(false);
        }
        ValueRecord compile_binary_expr(const ast::BinaryExpression &e) {
            auto binop = e->cast<ast::BinaryExpression>();
            auto op = binop->op;
            auto lhs = compile_expr(binop->lhs);
            auto rhs = compile_expr(binop->rhs);
            int prec_left, prec_right;
            prec_left = prec_right = std::numeric_limits<int>::max();
            if (binop->lhs->isa<ast::BinaryExpression>()) {
                prec_left = prec.opPrec[binop->lhs->cast<ast::BinaryExpression>()->op];
            }
            if (binop->rhs->isa<ast::BinaryExpression>()) {
                prec_right = prec.opPrec[binop->rhs->cast<ast::BinaryExpression>()->op];
            }
            auto [ty, L, R] = check_binary_expr(op, e->loc, lhs, rhs);
            Twine s, left, right;
            left = L.value;
            right = R.value;
            auto this_prec = prec.opPrec[e->op];
            if (this_prec > prec_left) {
                left = Twine::concat("(", left).append(")");
            }
            if (this_prec > prec_right) {
                right = Twine::concat("(", right).append(")");
            }
            return {Twine::concat(left, " " + op + " ", right), ty};
        }
        ValueRecord compile_ctor_call(const ast::ConstructorCall &call) {
            auto type = process_type(call->type);
            auto ctor_name = type_to_str(type);
            std::vector<ValueRecord> args;
            for (int i = 0; i < call->args.size(); i++) {
                auto arg = compile_expr(call->args[i]);
                args.emplace_back(arg);
            }
            Twine s(ctor_name + "(");
            for (int i = 0; i < args.size(); i++) {
                s.append(args[i].value);
                if (i + 1 < args.size()) {
                    s.append(",");
                }
            }
            s.append(")");
            return {s, type};
        }
        ValueRecord compile_func_call(const ast::FunctionCall &call) {
            auto func = call->func->identifier;

            std::vector<ValueRecord> args;
            std::vector<type::Type> arg_types;
            for (int i = 0; i < call->args.size(); i++) {
                auto arg = compile_expr(call->args[i]);
                // auto &arg_ty = arg.type;
                args.emplace_back(arg);
                arg_types.emplace_back(arg.type);
            }
            auto mangled_name = Mangler().mangle(func, arg_types);
            if (prototypes.find(func) == prototypes.end()) {
                error(call->loc, fmt::format("no function named", func));
            }
            auto &rec = prototypes.at(func);
            if (rec.overloads.find(mangled_name) == rec.overloads.end()) {
                std::string err;
                for (auto &a : arg_types) {
                    err.append(fmt::format("{}, ", type_to_str(a)));
                }
                error(call->loc, fmt::format("not matching function call to {} with args ({})", func, err));
            }
            Twine s(func + "(");
            for (int i = 0; i < args.size(); i++) {
                s.append(args[i].value);
                if (i + 1 < args.size()) {
                    s.append(",");
                }
            }
            s.append(")");
            return {s, rec.overloads.at(mangled_name)->ret};
        }
        ValueRecord compile_member_access(const ast::MemberAccess &access) {
            auto agg = compile_expr(access->var);
            if (agg.type->isa<type::VectorType>()) {
                auto v = agg.type->cast<type::VectorType>();
                auto member = access->member;
                return {agg.value.append("." + member), v->element_type};
            }
            AKR_ASSERT(false);
        }

        ValueRecord compile_expr(const ast::Expr &e) {
            if (e->isa<ast::Literal>()) {
                return compile_literal(e->cast<ast::Literal>());
            }
            if (e->isa<ast::Identifier>()) {
                return compile_var(e->cast<ast::Identifier>());
            }
            if (e->isa<ast::BinaryExpression>()) {
                return compile_binary_expr(e->cast<ast::BinaryExpression>());
            }
            if (e->isa<ast::FunctionCall>()) {
                return compile_func_call(e->cast<ast::FunctionCall>());
            }
            if (e->isa<ast::ConstructorCall>()) {
                return compile_ctor_call(e->cast<ast::ConstructorCall>());
            }
            if (e->isa<ast::MemberAccess>()) {
                return compile_member_access(e->cast<ast::MemberAccess>());
            }
            AKR_ASSERT(false);
        }
        template <class F>
        void auto_indent(ast::Stmt st, F &&f) {
            if (st->isa<ast::SeqStmt>()) {
                f();
            } else
                with_block([&] { f(); });
        }
        void compile_if(std::ostringstream &os, const ast::IfStmt &st) {
            auto cond = compile_expr(st->cond);
            if (cond.type != type::boolean) {
                error(st->cond->loc, "if cond must be boolean expression");
            }
            wl(os, "if({})", cond.value.str());
            auto_indent(st->if_true, [&] { compile_stmt(os, st->if_true); });
            if (st->if_false) {
                wl(os, "else");
                auto_indent(st->if_false, [&] { compile_stmt(os, st->if_false); });
            }
        }
        void compile_while(std::ostringstream &os, const ast::WhileStmt &st) {
            auto cond = compile_expr(st->cond);
            if (cond.type != type::boolean) {
                error(st->cond->loc,
                      fmt::format("while cond must be boolean expression but have {}", type_to_str(cond.type)));
            }
            wl(os, "while({})", cond.value.str());
            auto_indent(st->body, [&] {
                loop_pred = true;
                compile_stmt(os, st->body);
                loop_pred = false;
            });
        }
        void compile_var_decl(std::ostringstream &os, const ast::VarDeclStmt &stmt) {
            auto decl = stmt->decl;
            return compile_var_decl(os, decl);
        }
        void compile_var_decl(std::ostringstream &os, const ast::VarDecl &decl) {
            auto ty = process_type(decl->type);
            Twine s(type_to_str(ty) + " " + decl->var->identifier);
            if (decl->init) {
                auto init = compile_expr(decl->init);
                s.append(" = ").append(init.value);
            }
            wl(os, "{};", s.str());
            vars.insert(decl->var->identifier, {decl->var->identifier, ty});
        }
        void compile_assignment(std::ostringstream &os, const ast::Assignment &asgn) {
            auto lvalue = compile_expr(asgn->lhs);
            auto rvalue = compile_expr(asgn->rhs);
            wl(os, "{} {} {};", lvalue.value.str(), asgn->op, rvalue.value.str());
        }
        void compile_ret(std::ostringstream &os, const ast::Return &ret) {
            auto r = compile_expr(ret->expr);
            wl(os, "return {};", r.value.str());
        }
        void compile_for(std::ostringstream &os, const ast::ForStmt &st) {}
        void compile_stmt(std::ostringstream &os, const ast::Stmt &stmt) {
            if (stmt->isa<ast::VarDeclStmt>()) {
                compile_var_decl(os, stmt->cast<ast::VarDeclStmt>());
            } else if (stmt->isa<ast::Assignment>()) {
                compile_assignment(os, stmt->cast<ast::Assignment>());
            } else if (stmt->isa<ast::Return>()) {
                compile_ret(os, stmt->cast<ast::Return>());
            } else if (stmt->isa<ast::SeqStmt>()) {
                compile_block(os, stmt->cast<ast::SeqStmt>());
            } else if (stmt->isa<ast::IfStmt>()) {
                compile_if(os, stmt->cast<ast::IfStmt>());
            } else if (stmt->isa<ast::WhileStmt>()) {
                compile_while(os, stmt->cast<ast::WhileStmt>());
            } else if (stmt->isa<ast::ForStmt>()) {
                compile_for(os, stmt->cast<ast::ForStmt>());
            } else if (stmt->isa<ast::BreakStmt>()) {
                if (!loop_pred) {
                    error(stmt->loc, "`continue` outside of loop!");
                }
                wl(os, "break;");
            } else if (stmt->isa<ast::ContinueStmt>()) {
                if (!loop_pred) {
                    error(stmt->loc, "`continue` outside of loop!");
                }
                wl(os, "continue;");
            } else {
                AKR_ASSERT(false);
            }
        }
        void compile_block(std::ostringstream &os, const ast::SeqStmt &stmt) {
            wl(os, "{{");
            with_block([&] {
                auto _ = vars.push();
                for (auto &s : stmt->stmts) {
                    compile_stmt(os, s);
                }
            });
            wl(os, "}}");
        }
        Twine gen_func_prototype(const ast::FunctionDecl &func, bool is_def) {
            auto f_ty = process_type(func)->cast<type::FunctionType>();
            Twine s(type_to_str(f_ty->ret));
            s.append(" ").append(func->name->identifier).append("(");
            int cnt = 0;
            for (auto p : func->parameters) {
                auto ty = p->type;
                auto name = p->var->identifier;
                s.append(fmt::format("{} {}", type_to_str(process_type(ty)), name));
                if (size_t(cnt + 1) < func->parameters.size()) {
                    s.append(", ");
                }
                if (is_def) {
                    vars.insert(name, {name, process_type(ty)});
                }
            }
            s.append(")");
            return s;
        }
        void compile_func(std::ostringstream &os, const ast::FunctionDecl &func) {
            auto _ = vars.push();
            auto s = gen_func_prototype(func, func->body != nullptr);
            if (is_cuda) {
                s = Twine::concat("__host__ __device__ ", s);
            }
            if (func->body) {
                wl(os, "{}", s.str());
                compile_block(os, func->body);
            } else {
                wl(os, "{};", s.str());
            }
        }

      public:
        virtual std::string do_generate() {
            std::ostringstream os;
            wl(os, "namespace akari::asl {{");
            with_block([&]() {
                wl(os, "template<class C>");
                wl(os, "class {} {{", module.name);
                with_block([&]() {
                    wl(os, "public:");
                    wl(os, "AKR_IMPORT_TYPES()");
                    for (auto &unit : module.translation_units) {
                        for (auto &func : unit->funcs) {
                            compile_func(os, func);
                        }
                    }
                });
                wl(os, "}};");
            });
            wl(os, "}};");
            return os.str();
        }
    };
    std::unique_ptr<CodeGenerator> cpp_generator() { return std::make_unique<CodeGenCPP>(false); }
    std::unique_ptr<CodeGenerator> cuda_generator() { return std::make_unique<CodeGenCPP>(true); }
} // namespace akari::asl