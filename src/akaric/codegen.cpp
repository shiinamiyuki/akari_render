
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
        types["void"] = type::void_;
        types["bool"] = type::boolean;
        types["int"] = type::int32;
        types["uint"] = type::uint32;
        types["double"] = type::float64;
        types["float"] = type::float32;
        for (int i = 2; i <= 4; i++) {
            types[fmt::format("vec{}", i)] = create_vec_type(type::float32, i);
            types[fmt::format("ivec{}", i)] = create_vec_type(type::int32, i);
            types[fmt::format("uvec{}", i)] = create_vec_type(type::uint32, i);
            types[fmt::format("bvec{}", i)] = create_vec_type(type::boolean, i);
            types[fmt::format("dvec{}", i)] = create_vec_type(type::float64, i);
        }
    }
    type::Qualifier process_qualifier(const ast::TypeDecl &decl) { return decl->qualifier; }

    type::AnnotatedType CodeGenerator::process_type(const ast::AST &n) {
        if (n->isa<ast::VarDecl>()) {
            return process_type(n->cast<ast::VarDecl>()->type);
        }
        if (n->isa<ast::Typename>()) {
            auto ty = n->cast<ast::Typename>();
            if (types.find(ty->name) == types.end()) {
                throw std::runtime_error(fmt::format("definition of type {} not found", ty->name));
            }
            auto t = types.at(ty->name);
            return type::AnnotatedType{t, process_qualifier(ty)};
        }
        if (n->isa<ast::StructDecl>()) {
            return type::AnnotatedType{process_struct_decl(n->cast<ast::StructDecl>()),
                                       process_qualifier(n->cast<ast::StructDecl>())};
        }
        if (n->isa<ast::ArrayDecl>()) {
            auto ty = n->cast<ast::ArrayDecl>();
            auto e = process_type(ty->element_type);
            if (e.qualifier != type::Qualifier::none) {
                error(ty->loc, "array element type cannot have qualifiers");
            }
            int length = -1;
            if (ty->length) {
                length = eval_const_int(ty->length);
            }
            return type::AnnotatedType(type_ctx.make_array(e.type, length), process_qualifier(ty));
        }
        if (n->isa<ast::TupleDecl>()) {
            auto ty = n->cast<ast::TupleDecl>();
            std::vector<type::Type> elements;
            for (auto e : ty->elements) {
                auto et = process_type(e);
                if (et.qualifier != type::Qualifier::none) {
                    error(ty->loc, "tuple element type cannot have qualifiers");
                }
                elements.emplace_back(et.type);
            }
            return type::AnnotatedType(type_ctx.make_tuple(elements), process_qualifier(ty));
        }
        if (n->isa<ast::FunctionDecl>()) {
            auto func = n->cast<ast::FunctionDecl>();
            auto ty = std::make_shared<type::FunctionTypeNode>();
            for (auto &a : func->parameters) {
                ty->args.emplace_back(process_type(a->type).type);
            }
            auto ret = process_type(func->type);
            ty->ret = ret.type;
            return ty;
        }
        AKR_ASSERT(false);
    }
    type::StructType CodeGenerator::process_struct_decl(const ast::StructDecl &decl) {
        if (structs.find(decl->struct_name->name) != structs.end()) {
            return structs.at(decl->struct_name->name);
        }
        auto st = std::make_shared<type::StructTypeNode>();
        for (auto &field : decl->fields) {
            auto t = process_type(field->type);
            type::StructField f;
            f.index = (int)st->fields.size();
            f.name = field->var->identifier;
            f.type = t.type;
            st->fields.emplace_back(f);
        }
        st->name = decl->struct_name->name;
        types.emplace(st->name, st);
        structs[st->name] = st;
        return st;
    }
    std::string CodeGenerator::generate(const BuildConfig &config_, const Module &module_) {
        this->config = config_;
        this->module = module_;
        add_type_parameters();
        for (auto &unit : module.translation_units) {
            for (auto &def : unit->defs) {
                if (def->isa<ast::StructDecl>()) {
                    (void)process_struct_decl(def->cast<ast::StructDecl>());
                } else if (def->isa<ast::ConstDecl>()) {
                    auto decl = def->cast<ast::ConstDecl>();
                    ValueRecord v =
                        ValueRecord{decl->var->var->identifier, type::AnnotatedType(process_type(decl->var->type))};
                    vars.insert(decl->var->var->identifier, v);
                    if (v.type() == type::int32 || v.type() == type::uint32) {

                        const_ints.insert(decl->var->var->identifier, eval_const_int(decl->var->init));
                    }
                } else if (def->isa<ast::BufferObject>()) {
                    auto decl = def->cast<ast::BufferObject>();
                    ValueRecord v =
                        ValueRecord{decl->var->var->identifier, type::AnnotatedType(process_type(decl->var->type))};
                    vars.insert(decl->var->var->identifier, v);
                } else if (def->isa<ast::UniformVar>()) {
                    auto decl = def->cast<ast::UniformVar>();
                    ValueRecord v =
                        ValueRecord{decl->var->var->identifier, type::AnnotatedType(process_type(decl->var->type))};
                    vars.insert(decl->var->var->identifier, v);
                }
            }
        }
        process_prototypes();

        return do_generate();
    }

    void CodeGenerator::process_prototypes() {
        for (auto &unit : module.translation_units) {
            for (auto &decl : unit->funcs) {
                auto f_ty = process_type(decl).type->cast<type::FunctionType>();
                prototypes[decl->name->identifier].overloads.emplace(
                    Mangler().mangle(decl->name->identifier, f_ty->args),
                    FunctionRecord::Entry(f_ty, decl->is_intrinsic));
            }
        }
    }
    int CodeGenerator::eval_const_int(const ast::Expr &e) {
        if (e->isa<ast::IntLiteral>()) {
            return e->cast<ast::IntLiteral>()->val;
        }
        if (e->isa<ast::Identifier>()) {
            auto var = e->cast<ast::Identifier>()->identifier;
            if (auto i = const_ints.at(var)) {
                return i.value();
            } else {
                error(e->loc, fmt::format("{} is not a const int expression", var));
            }
        } else if (e->isa<ast::BinaryExpression>()) {
            auto binexp = e->cast<ast::BinaryExpression>();
            auto lhs = eval_const_int(binexp->lhs);
            auto rhs = eval_const_int(binexp->rhs);
            auto op = binexp->op;
            if (op == "+") {
                return lhs + rhs;
            }
            if (op == "-") {
                return lhs - +rhs;
            } else {
                error(e->loc, fmt::format("operator {} is not supported in const int expression", op));
            }
        } else {
            error(e->loc, fmt::format("operation not supported in const int expression"));
        }
    }
    class CodeGenCPP : public CodeGenerator {
        bool loop_pred = false;
        bool is_cuda;
        OperatorPrecedence prec;
        std::unordered_map<type::Type, std::string> type_mangler_cached;

      public:
        CodeGenCPP(bool is_cuda) : is_cuda(is_cuda) {}
        std::string type_mangler(const type::Type &ty) {
            if (type_mangler_cached.find(ty) != type_mangler_cached.end()) {
                return type_mangler_cached.at(ty);
            }
            auto s = type_mangler_do(ty);
            type_mangler_cached[ty] = s;
            return s;
        }
        std::string type_mangler_do(const type::Type &ty) {
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
                    return fmt::format("{}{}", type_mangler(et), n);
                }
            }
            if (ty->isa<type::StructType>()) {
                return "struct_" + ty->cast<type::StructType>()->name;
            }
            if (ty->isa<type::ArrayType>()) {
                auto arr = ty->cast<type::ArrayType>();
                if (arr->length == -1) {
                    return fmt::format("DArray_", type_mangler(arr->element_type));
                } else {
                    return fmt::format("Array_{}_{}", type_mangler(arr->element_type), arr->length);
                }
            }
            if (ty->isa<type::TupleType>()) {
                auto tuple = ty->cast<type::TupleType>();
                Twine s("Tuple");
                for (auto &e : tuple->element_types) {
                    s.append("_" + type_mangler(e));
                }
                auto m = s.str();
                CodeBlock block;
                {
                    block.wl("struct {} {{", m);
                    int cnt = 0;
                    for (auto &e : tuple->element_types) {
                        block.wl("    {} _{};", _type_to_str(e), cnt++);
                    }
                    block.wl("}};");
                };
                misc_defs.emplace_back(std::move(block));
                return m;
            }
            AKR_ASSERT(false);
        }
        std::string _type_to_str(const type::Type &ty) {
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
            if (ty->isa<type::ArrayType>()) {
                auto arr = ty->cast<type::ArrayType>();
                if (arr->length == -1) {
                    return fmt::format("{}*", type_to_str(arr->element_type));
                } else {
                    return fmt::format("astd::array<{}, {}>", type_to_str(arr->element_type), arr->length);
                }
            }
            if (ty->isa<type::TupleType>()) {
                auto tuple = ty->cast<type::TupleType>();
                auto mangled = type_mangler(tuple);
                return mangled;
            }
            if (ty->isa<type::VoidType>()) {
                return "void";
            }
            AKR_ASSERT(false);
        }

        std::string type_to_str(const type::AnnotatedType &anno) {
            auto ty = anno.type;
            auto s = _type_to_str(ty);
            if (anno.qualifier & type::Qualifier::out) {
                s = s + " &";
            }
            if (anno.qualifier & type::Qualifier::const_) {
                s = "const " + s;
            }
            return s;
        }

        ValueRecord compile_var(const ast::Identifier &var) {
            if (!vars.at(var->identifier).has_value()) {
                error(var->loc, fmt::format("identifier {} not found", var->identifier));
            }
            auto r = vars.at(var->identifier).value();
            return {var->identifier, r.annotated_type};
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
            if (lit->isa<ast::BoolLiteral>()) {
                auto i = lit->cast<ast::BoolLiteral>();
                return {fmt::format("{}", i->val ? "true" : "false"), type::boolean};
            }
            AKR_ASSERT(false);
        }

        // type::Type check_binary_expr(const std::string &op, const type::Type &lhs, const type::Type &rhs) {

        //     if (op == "<" || op == "<=" || op == ">" || op == ">=" || op == "!=" || op == "==") {
        //         if (lhs->isa<type::VectorType>() || rhs->isa<type::VectorType>()) {
        //             return nullptr;
        //         }
        //         return type::boolean;
        //     }
        //     if (op == "+" || op == "-" || op == "*" || op == "/") {
        //         if()
        //     }
        // }
        std::tuple<type::Type, ValueRecord, ValueRecord> check_binary_expr(const std::string &op,
                                                                           const SourceLocation &loc,
                                                                           const ValueRecord &lhs,
                                                                           const ValueRecord &rhs) {
            if (op == "||" || op == "&&") {
                if (lhs.type() != type::boolean || rhs.type() != type::boolean) {
                    error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type()->type_name(),
                                           rhs.type()->type_name()));
                }
                return std::make_tuple(type::boolean, lhs, rhs);
            }

            if (op == "+" || op == "-" || op == "*" || op == "/") {
                if (lhs.type() == rhs.type()) {
                    return std::make_tuple(lhs.type(), lhs, rhs);
                }
                if (lhs.type()->isa<type::VectorType>() && rhs.type()->isa<type::PrimitiveType>()) {
                    auto vt = lhs.type()->cast<type::VectorType>();
                    auto et = vt->element_type;
                    if (et != rhs.type()) {
                        error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type()->type_name(),
                                               rhs.type()->type_name()));
                    }
                    return std::make_tuple(lhs.type(), lhs, ValueRecord{rhs.value, lhs.type()});
                } else if (rhs.type()->isa<type::VectorType>() && lhs.type()->isa<type::PrimitiveType>()) {
                    auto vt = rhs.type()->cast<type::VectorType>();
                    auto et = vt->element_type;
                    if (et != lhs.type()) {
                        error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type()->type_name(),
                                               rhs.type()->type_name()));
                    }
                    return std::make_tuple(rhs.type(), ValueRecord{lhs.value, rhs.type()}, rhs);
                } else {
                    error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type()->type_name(),
                                           rhs.type()->type_name()));
                }
            } else if (op == "<" || op == "<=" || op == ">" || op == ">=" || op == "!=" || op == "==") {
                if (lhs.type()->isa<type::VectorType>() || rhs.type()->isa<type::VectorType>()) {
                    error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type()->type_name(),
                                           rhs.type()->type_name()));
                }
                if (lhs.type() != rhs.type()) {
                    error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type()->type_name(),
                                           rhs.type()->type_name()));
                }
                return std::make_tuple(type::boolean, lhs, rhs);
            }

            AKR_ASSERT(false);
        }
        ValueRecord compile_cond_expr(const ast::ConditionalExpression &e) {
            auto cond = compile_expr(e->cond);
            if (cond.type() != type::boolean) {
                error(e->cond->loc, fmt::format("conditional expression must be boolean expression but have {}",
                                                type_to_str(cond.type())));
            }
            auto lhs = compile_expr(e->lhs);
            auto rhs = compile_expr(e->rhs);
            if (lhs.type() != rhs.type()) {
                error(e->cond->loc, fmt::format("conditional expression must have the same type but have {} and {}",
                                                type_to_str(lhs.type()), type_to_str(rhs.type())));
            }
            return ValueRecord{cond.value.append(" ? ").append(lhs.value).append(" : ").append(rhs.value),
                               lhs.annotated_type};
        }
        ValueRecord compile_unary_expr(const ast::UnaryExpression &e) {
            auto v = compile_expr(e->operand);
            return ValueRecord{Twine(e->op).append(v.value), v.annotated_type};
        }
        ValueRecord compile_binary_expr(const ast::BinaryExpression &e) {
            static std::unordered_map<std::string, std::string> op2func = {
                {"+", "__add__"},
                {"-", "__sub__"},
                {"*", "__mul__"},
                {"/", "__div__"},
            };
            auto binop = e->cast<ast::BinaryExpression>();
            auto op = binop->op;
            auto lhs = compile_expr(binop->lhs);
            auto rhs = compile_expr(binop->rhs);

            // check if overloading presents
            if (op2func.find(op) != op2func.end()) {
                auto func = op2func.at(op);
                auto mangled_func = Mangler().mangle(func, {lhs.type(), rhs.type()});
                if (prototypes.find(func) != prototypes.end()) {
                    auto &proto = prototypes.at(func);
                    for (auto &overload : proto.overloads) {
                        if (overload.first == mangled_func) {
                            return {
                                Twine(func).append("(").append(lhs.value).append(" ,").append(rhs.value).append(")"),
                                overload.second.type->ret};
                        }
                    }
                }
            }

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
            if (this_prec > prec_right || op == "/" || op == "%" || op == "-") {
                right = Twine::concat("(", right).append(")");
            }
            return {Twine::concat(left, " " + op + " ", right), ty};
        }
        ValueRecord compile_ctor_call(const ast::ConstructorCall &call) {
            auto type = process_type(call->type);
            auto ctor_name = type_to_str(type.type);
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
                arg_types.emplace_back(arg.type());
            }
            auto mangled_name = Mangler().mangle(func, arg_types);
            if (prototypes.find(func) == prototypes.end()) {
                error(call->loc, fmt::format("no function named {}", func));
            }
            auto &rec = prototypes.at(func);
            if (rec.overloads.find(mangled_name) == rec.overloads.end()) {
                std::string err;
                for (auto &a : arg_types) {
                    err.append(fmt::format("{}, ", type_to_str(a)));
                }
                error(call->loc, fmt::format("not matching function call to {} with args ({})", func, err));
            }
            // auto &overload = rec.overloads.at(mangled_name);
            // for (uint32_t i = 0; i < arg_types.size(); i++) {
            // }
            Twine s(func + "(");
            for (int i = 0; i < args.size(); i++) {
                s.append(args[i].value);
                if (i + 1 < args.size()) {
                    s.append(",");
                }
            }
            s.append(")");
            return {s, rec.overloads.at(mangled_name).type->ret};
        }
        ValueRecord compile_index(const ast::Index &idx) {
            auto agg = compile_expr(idx->expr);
            if (!agg.type()->isa<type::ArrayType>()) {
                error(idx->expr->loc, "operator [] only allowed for arrays");
            }
            return ValueRecord{agg.value.append("[").append(compile_expr(idx->idx).value).append("]"),
                               agg.type()->cast<type::ArrayType>()->element_type};
        }
        ValueRecord compile_tuple(const ast::TupleExpression &tuple) {
            std::vector<ValueRecord> elements;
            std::vector<type::Type> _types;
            for (auto e : tuple->expr) {
                auto v = compile_expr(e);
                elements.emplace_back(v);
                _types.emplace_back(v.type());
            }
            auto tuple_type = type_ctx.make_tuple(_types);
            Twine s(_type_to_str(tuple_type));
            s.append("{");
            for (auto &e : elements) {
                s.append(e.value).append(", ");
            }
            s.append("}");
            return ValueRecord{s, type::AnnotatedType(tuple_type)};
        }
        ValueRecord compile_member_access(const ast::TupleAccess &access) {
            auto agg = compile_expr(access->var);
            if (agg.type()->isa<type::TupleType>()) {
                auto t = agg.type()->cast<type::TupleType>();
                if ((size_t)access->member < t->element_types.size()) {
                    return {agg.value.append("." + std::to_string(access->member)),
                            type::AnnotatedType(t->element_types[access->member], agg.annotated_type.qualifier)};
                } else {
                    error(access->loc, fmt::format("tuple type does not have member {}", access->member));
                }
            } else {
                error(access->loc, "tuple type expected");
            }
        }
        ValueRecord compile_member_access(const ast::MemberAccess &access) {
            auto agg = compile_expr(access->var);
            if (agg.type()->isa<type::VectorType>()) {
                auto v = agg.type()->cast<type::VectorType>();
                auto member = access->member;
                return {agg.value.append("." + member),
                        type::AnnotatedType(v->element_type, agg.annotated_type.qualifier)};
            } else if (agg.type()->isa<type::StructType>()) {
                auto st = agg.type()->cast<type::StructType>();
                auto member = access->member;
                auto it = std::find_if(st->fields.begin(), st->fields.end(),
                                       [&](auto &field) { return field.name == member; });
                if (it == st->fields.end()) {
                    error(access->loc, fmt::format("type {} does not have member {}", st->name, member));
                }
                return {agg.value.append("." + member), type::AnnotatedType(it->type, agg.annotated_type.qualifier)};
            } else if (agg.type()->isa<type::ArrayType>()) {
                auto arr = agg.type()->cast<type::ArrayType>();
                auto member = access->member;
                if (member != "length") {
                    error(access->loc, fmt::format("array type does not have member {}", member));
                } else {
                    if (arr->length >= 0) {
                        return {std::to_string(arr->length), type::uint32};
                    } else {
                        return {agg.value.append(".size()"), type::uint32};
                    }
                }
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
            if (e->isa<ast::UnaryExpression>()) {
                return compile_unary_expr(e->cast<ast::UnaryExpression>());
            }
            if (e->isa<ast::ConditionalExpression>()) {
                return compile_cond_expr(e->cast<ast::ConditionalExpression>());
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
            if (e->isa<ast::TupleAccess>()) {
                return compile_member_access(e->cast<ast::TupleAccess>());
            }
            if (e->isa<ast::Index>()) {
                return compile_index(e->cast<ast::Index>());
            }
            if (e->isa<ast::TupleExpression>()) {
                return compile_tuple(e->cast<ast::TupleExpression>());
            }
            AKR_ASSERT(false);
        }
        template <class F>
        void auto_indent(CodeBlock &block, ast::Stmt st, F &&f) {
            if (st->isa<ast::SeqStmt>()) {
                f();
            } else
                block.with_block([&] { f(); });
        }
        void compile_if(CodeBlock &block, const ast::IfStmt &st) {
            auto cond = compile_expr(st->cond);
            if (cond.type() != type::boolean) {
                error(st->cond->loc, "if cond must be boolean expression");
            }
            block.wl("if({})", cond.value.str());
            auto_indent(block, st->if_true, [&] { compile_stmt(block, st->if_true); });
            if (st->if_false) {
                block.wl("else");
                auto_indent(block, st->if_false, [&] { compile_stmt(block, st->if_false); });
            }
        }
        void compile_while(CodeBlock &block, const ast::WhileStmt &st) {
            auto cond = compile_expr(st->cond);
            if (cond.type() != type::boolean) {
                error(st->cond->loc,
                      fmt::format("while cond must be boolean expression but have {}", type_to_str(cond.type())));
            }
            block.wl("while({})", cond.value.str());
            auto_indent(block, st->body, [&] {
                loop_pred = true;
                compile_stmt(block, st->body);
                loop_pred = false;
            });
        }
        void compile_var_decl(CodeBlock &block, const ast::VarDeclStmt &stmt) {
            auto decl = stmt->decl;
            return compile_var_decl(block, decl);
        }
        void compile_var_decl(CodeBlock &block, const ast::LetDeclStmt &stmt) {
            auto init = compile_expr(stmt->init);
            type::Type ty;
            if (stmt->type) {
                auto expected = process_type(stmt->type).type;
                if (expected != init.type()) {
                    error(stmt->var->loc,
                          fmt::format("{} expected but found {}", type_to_str(expected), type_to_str(init.type())));
                }
                ty = expected;
            } else {
                ty = init.type();
            }
            block.wl("{} {} = {};", type_to_str(ty), stmt->var->identifier, init.value.str());
        }
        void compile_var_decl(CodeBlock &block, const ast::VarDecl &decl) {
            auto ty = process_type(decl->type);
            Twine s(type_to_str(ty) + " " + decl->var->identifier);
            if (decl->init) {
                auto init = compile_expr(decl->init);
                s.append(" = ").append(init.value);
            }
            block.wl("{};", s.str());
            if (vars.frame->at(decl->var->identifier)) {
                error(decl->loc, fmt::format("{} is already defined", decl->var->identifier));
            }
            vars.insert(decl->var->identifier, {decl->var->identifier, ty});
        }
        // void destructure(std::vector<std::string> lines, const ast::TupleExpression &lhs, const ast::Expr &rhs) {

        // }
        void compile_assignment(CodeBlock &block, const ast::Destructure &des) {
            // std::vector<std::string> lines;
            // destructure(lines, des->lhs, des->rhs);
            auto rhs = compile_expr(des->rhs);
            int tmp = temp_counter++;
            block.wl("{} __Gen_tmp{} = {};", type_to_str(rhs.annotated_type), tmp, rhs.value.str());
            auto vars_ = des->lhs;
            if (rhs.type()->isa<type::StructType>()) {
                auto st = rhs.type()->cast<type::StructType>();
                if (vars_->expr.size() != st->fields.size()) {
                    error(des->loc,
                          fmt::format("RHS has {} components but LHS has {}", st->fields.size(), vars_->expr.size()));
                }
                for (size_t i = 0; i < st->fields.size(); i++) {
                    if (!vars_->expr[i]->isa<ast::Identifier>()) {
                        error(vars_->loc, "nested destructuring is not allowed");
                    }
                    auto var = vars_->expr[i]->cast<ast::Identifier>();
                    block.wl("{} {} = __Gen_tmp{}.{};", type_to_str(st->fields[i].type), var->identifier, temp_counter,
                             st->fields[i].name);
                }
            } else if (rhs.type()->isa<type::TupleType>()) {
                auto st = rhs.type()->cast<type::TupleType>();
                if (vars_->expr.size() != st->element_types.size()) {
                    error(des->loc, fmt::format("RHS has {} components but LHS has {}", st->element_types.size(),
                                                vars_->expr.size()));
                }
                for (size_t i = 0; i < st->element_types.size(); i++) {
                    if (!vars_->expr[i]->isa<ast::Identifier>()) {
                        error(vars_->loc, "nested destructuring is not allowed");
                    }
                    auto var = vars_->expr[i]->cast<ast::Identifier>();
                    block.wl("{} {} = __Gen_tmp{}._{};", type_to_str(st->element_types[i]), var->identifier, temp_counter, i);
                }
            } else {
                error(des->rhs->loc, "only destructuring of struct or tuple is allowed");
            }
        }
        void compile_assignment(CodeBlock &block, const ast::Assignment &asgn) {
            auto lvalue = compile_expr(asgn->lhs);
            if ((int)lvalue.annotated_type.qualifier & (int)type::Qualifier::const_) {
                error(asgn->loc, fmt::format("cannot assign to const value"));
            }
            auto rvalue = compile_expr(asgn->rhs);
            block.wl("{} {} {};", lvalue.value.str(), asgn->op, rvalue.value.str());
        }
        void compile_call_stmt(CodeBlock &block, const ast::CallStmt &call) {
            block.wl("{};", compile_expr(call->call).value.str());
        }
        void compile_ret(CodeBlock &block, const ast::Return &ret) {
            auto r = compile_expr(ret->expr);
            block.wl("return {};", r.value.str());
        }
        void compile_switch(CodeBlock &block, const ast::SwitchStmt &st) {
            auto cond = compile_expr(st->cond);
            if (!cond.type()->is_int()) {
                error(st->cond->loc,
                      fmt::format("switch cond must be int expression but have {}", type_to_str(cond.type())));
            }
            block.wl("switch({})", cond.value.str());
            block.wl("{{");
            for (auto c : st->cases) {
                for (auto &label : c.first) {
                    if (label.is_default) {
                        block.wl("default:");
                    } else
                        block.wl("case {}:", compile_expr(label.value).value.str());
                    compile_block(block, c.second);
                }
            }
            block.wl("}}");
        }
        void compile_for(CodeBlock &block, const ast::ForStmt &st) {
            block.wl("{{ // for begin");
            with_block([&] {
                compile_var_decl(block, st->init);
                auto cond = compile_expr(st->cond);
                if (cond.type() != type::boolean) {
                    error(st->cond->loc,
                          fmt::format("while cond must be boolean expression but have {}", type_to_str(cond.type())));
                }
                block.wl("while({})", cond.value.str());
                auto_indent(block, st->body, [&] {
                    compile_stmt(block, st->body);
                    compile_stmt(block, st->step);
                });
            });
            block.wl("}} // for end");
        }
        void compile_stmt(CodeBlock &block, const ast::Stmt &stmt) {
            if (stmt->isa<ast::VarDeclStmt>()) {
                compile_var_decl(block, stmt->cast<ast::VarDeclStmt>());
            } else if (stmt->isa<ast::LetDeclStmt>()) {
                compile_var_decl(block, stmt->cast<ast::LetDeclStmt>());
            } else if (stmt->isa<ast::Assignment>()) {
                compile_assignment(block, stmt->cast<ast::Assignment>());
            } else if (stmt->isa<ast::Destructure>()) {
                compile_assignment(block, stmt->cast<ast::Destructure>());
            } else if (stmt->isa<ast::CallStmt>()) {
                compile_call_stmt(block, stmt->cast<ast::CallStmt>());
            } else if (stmt->isa<ast::Return>()) {
                compile_ret(block, stmt->cast<ast::Return>());
            } else if (stmt->isa<ast::SeqStmt>()) {
                compile_block(block, stmt->cast<ast::SeqStmt>());
            } else if (stmt->isa<ast::IfStmt>()) {
                compile_if(block, stmt->cast<ast::IfStmt>());
            } else if (stmt->isa<ast::WhileStmt>()) {
                compile_while(block, stmt->cast<ast::WhileStmt>());
            } else if (stmt->isa<ast::ForStmt>()) {
                compile_for(block, stmt->cast<ast::ForStmt>());
            } else if (stmt->isa<ast::SwitchStmt>()) {
                compile_switch(block, stmt->cast<ast::SwitchStmt>());
            } else if (stmt->isa<ast::BreakStmt>()) {
                if (!loop_pred) {
                    error(stmt->loc, "`continue` outside of loop!");
                }
                block.wl("break;");
            } else if (stmt->isa<ast::ContinueStmt>()) {
                if (!loop_pred) {
                    error(stmt->loc, "`continue` outside of loop!");
                }
                block.wl("continue;");
            } else {
                AKR_ASSERT(false);
            }
        }
        void compile_block(CodeBlock &block, const ast::SeqStmt &stmt) {
            block.wl("{{");
            block.with_block([&] {
                auto _ = vars.push();
                for (auto &s : stmt->stmts) {
                    compile_stmt(block, s);
                }
            });
            block.wl("}}");
        }
        Twine gen_func_prototype(const ast::FunctionDecl &func, bool is_def) {
            auto f_ty = process_type(func).type->cast<type::FunctionType>();
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
                cnt++;
            }
            s.append(")");
            return s;
        }
        CodeBlock compile_buffer(const ast::BufferObject &buf) {
            CodeBlock block;
            auto type = process_type(buf->var->type).type;
            if (type->isa<type::ArrayType>()) {
                auto arr = type->cast<type::ArrayType>();
                if (arr->length == -1) {
                    block.wl("Buffer<{}> {};", type_to_str(arr->element_type), buf->var->var->identifier);
                } else {
                    error(buf->loc, "buffer must be an T[]");
                }
            } else {
                error(buf->loc, "buffer must be an T[]");
            }
            return block;
        }
        CodeBlock compile_buffer_binder(const ast::BufferObject &buf) {
            CodeBlock block;
            auto type = process_type(buf->var->type).type;
            if (type->isa<type::ArrayType>()) {
                auto arr = type->cast<type::ArrayType>();
                if (arr->length == -1) {
                    block.wl("std::vector<{}> {};", type_to_str(arr->element_type), buf->var->var->identifier);
                } else {
                    error(buf->loc, "buffer must be an T[]");
                }
            } else {
                error(buf->loc, "buffer must be an T[]");
            }
            return block;
        }
        void compile_buffer_binder_bind(std::ostringstream &os, const ast::BufferObject &buf) {
            wl(os, "{}.copy(binder.{})", buf->var->var->identifier, buf->var->var->identifier);
        }
        CodeBlock compile_uniform(const ast::UniformVar &u) {
            auto type = process_type(u->var->type).type;
            CodeBlock block;
            block.wl("{} {};", type_to_str(type), u->var->var->identifier);
            return block;
        }
        CodeBlock compile_const(const ast::ConstVar &cst) {
            auto type = process_type(cst->var->type).type;
            CodeBlock block;
            block.wl("const {} {} = {};", type_to_str(type), cst->var->var->identifier,
                     compile_expr(cst->var->init).value.str());
            return block;
        }
        CodeBlock compile_struct(const ast::StructDecl &st) {
            CodeBlock block;
            block.wl("struct {} {{", st->struct_name->name);
            block.with_block([&] {
                for (auto &field : st->fields) {
                    block.wl("{} {};", type_to_str(process_type(field->type)), field->var->identifier);
                }
            });
            block.wl("}};");
            return block;
        }
        CodeBlock compile_func(const ast::FunctionDecl &func) {
            if (func->is_intrinsic) {
                return {};
            }
            CodeBlock block;
            auto _ = vars.push();
            auto s = gen_func_prototype(func, func->body != nullptr);
            if (is_cuda) {
                s = Twine::concat("AKR_XPU ", s);
            }
            s = Twine::concat("inline ", s);
            if (func->body) {
                block.wl("{}", s.str());
                compile_block(block, func->body);
            } else {
                block.wl("{};", s.str());
            }
            return block;
        }

      public:
        virtual std::string do_generate() {
            std::ostringstream os;
            wl(os, "#pragma once");
            if (is_cuda) {
                wl(os, R"(#include <cuda.h>)");
            }
            wl(os, R"(#include <akari/common/color.h>)");
            wl(os, R"(#include <akari/common/buffer.h>)");
            wl(os, "namespace akari::asl {{");
            with_block([&]() {
                wl(os, "template<class C>");
                wl(os, "class {} {{", module.name);
                with_block([&]() {
                    wl(os, "public:");
                    wl(os, "AKR_IMPORT_TYPES()");
                    for (auto &unit : module.translation_units) {
                        for (auto &def : unit->defs) {
                            CodeBlock block;
                            if (def->isa<ast::UniformVar>()) {
                                block = compile_uniform(def->cast<ast::UniformVar>());
                            } else if (def->isa<ast::BufferObject>()) {
                                block = compile_buffer(def->cast<ast::BufferObject>());
                            } else if (def->isa<ast::ConstVar>()) {
                                block = compile_const(def->cast<ast::ConstVar>());
                            } else if (def->isa<ast::StructDecl>()) {
                                block = compile_struct(def->cast<ast::StructDecl>());
                            } else if (def->isa<ast::FunctionDecl>()) {
                                block = compile_func(def->cast<ast::FunctionDecl>());
                            }
                            for (auto &def_ : misc_defs) {
                                write(os, def_);
                            }
                            write(os, block);
                        }
                    }

                    wl(os, "struct BufferBinder {{");
                    with_block([&]() {
                        for (auto &unit : module.translation_units) {
                            for (auto &buf : unit->buffers) {
                                auto block = compile_buffer_binder(buf);
                                write(os, block);
                            }
                        }
                    });
                    wl(os, "}};");
                    wl(os, "void bind(const BufferBinder& binder){{");
                    with_block([&]() {
                        for (auto &unit : module.translation_units) {
                            for (auto &buf : unit->buffers) {
                                compile_buffer_binder_bind(os, buf);
                            }
                        }
                    });
                    wl(os, "}}");
                });
                wl(os, "}};");
            });
            wl(os, "}}");
            return os.str();
        }
    };
    std::unique_ptr<CodeGenerator> cpp_generator() { return std::make_unique<CodeGenCPP>(false); }
    std::unique_ptr<CodeGenerator> cuda_generator() { return std::make_unique<CodeGenCPP>(true); }
} // namespace akari::asl