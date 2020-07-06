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

#include <akari/asl/parser.h>
#include <akari/asl/backend.h>

#ifdef _MSC_VER
#pragma warning(disable : 4146)
#pragma warning(disable : 4624)
#pragma warning(disable : 4267)
#pragma warning(disable : 5030)
#pragma warning(disable : 4244)
#pragma warning(disable : 4324)
#pragma warning(disable : 4245)
#pragma warning(disable : 4458)
#pragma warning(disable : 4141)
#pragma warning(disable : 4459)
#endif

#include <llvm/ADT/STLExtras.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

namespace akari::asl {
    struct LLVMInit {
        LLVMInit() {
            llvm::InitializeNativeTarget();
            LLVMInitializeNativeAsmPrinter();
            LLVMInitializeNativeAsmParser();
        }
        ~LLVMInit() { llvm::llvm_shutdown(); }
    };
    static std::weak_ptr<LLVMInit> init;
    struct ValueRecord {
        llvm::Value *value;
        type::Type type;
    };
    struct LValueRecord {
        llvm::Value *value;
        type::Type type;
    };
    struct FunctionRecord {
        llvm::Function *function = nullptr;
        type::FunctionType type;
    };
    struct StructRecord {
        type::StructType type;
        llvm::StructType *llvm_type;
    };
    class LLVMBackend : public Backend {
        llvm::LLVMContext ctx;
        std::unique_ptr<llvm::Module> owner;
        std::shared_ptr<LLVMInit> _init;
        llvm::ExecutionEngine *EE = nullptr;
        FunctionRecord cur_function;
        std::unique_ptr<llvm::IRBuilder<>> builder;
        std::unordered_map<std::string, FunctionRecord> funcs;
        Environment<std::string, LValueRecord> vars;
        std::unordered_map<std::string, StructRecord> structs;
        std::unordered_map<std::string, type::Type> types;
        std::unordered_map<std::string, llvm::Function *> prototypes;
        std::unordered_map<type::Type, llvm::Type *> type_cache;
        [[noreturn]] void error(const SourceLocation &loc, std::string &&msg) {
            throw std::runtime_error(fmt::format("error: {} at {}:{}:{}", msg, loc.filename, loc.line, loc.col));
        }
        void init_types() {
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
            // for (int m = 2; m <= 4; m++) {
            //     for (int n = 2; n <= 4; n++) {
            //         types[fmt::format("mat{}", m, n)] = create_vec_type(type::float32, i);
            //         types[fmt::format("imat{}", m, n)] = create_vec_type(type::int32, i);
            //         types[fmt::format("bmat{}", m, n)] = create_vec_type(type::boolean, i);
            //         types[fmt::format("dmat{}", m, n)] = create_vec_type(type::float64, i);
            //     }
            // }
        }
        type::Type create_vec_type(const type::Type &base, int n) {
            auto v = std::make_shared<type::VectorTypeNode>();
            v->element_type = base;
            v->count = n;
            return v;
        }
        llvm::Type *to_llvm_type(type::Type ty) {
            if (ty->isa<type::PrimitiveType>()) {
                if (ty == type::boolean) {
                    return llvm::Type::getInt8Ty(ctx);
                }
                if (ty == type::int32) {
                    return llvm::Type::getInt32Ty(ctx);
                }
                if (ty == type::uint32) {
                    return llvm::Type::getInt32Ty(ctx);
                }
                if (ty == type::float32) {
                    return llvm::Type::getFloatTy(ctx);
                }
                if (ty == type::float64) {
                    return llvm::Type::getDoubleTy(ctx);
                }
                AKR_ASSERT(false);
            }
            if (ty->isa<type::StructType>()) {
                auto st = ty->cast<type::StructType>();
                AKR_ASSERT(structs.count(st->name));
                if (!structs.at(st->name).llvm_type) {
                    std::vector<llvm::Type *> fields;
                    for (auto &f : st->fields) {
                        fields.emplace_back(to_llvm_type(f.type));
                    }
                    auto res = llvm::StructType::create(ctx, fields, st->name);
                    structs[st->name] = StructRecord{st, res};
                    return res;
                }
                return structs.at(st->name).llvm_type;
            }
            if (ty->isa<type::VectorType>()) {
                auto v = ty->cast<type::VectorType>();
                return llvm::VectorType::get(to_llvm_type(v->element_type), v->count);
            }
            if (ty->isa<type::FunctionType>()) {
                if (type_cache.count(ty)) {
                    return type_cache.at(ty);
                }
                std::vector<llvm::Type *> args;
                llvm::Type *ret;
                auto func = ty->cast<type::FunctionType>();
                for (auto &a : func->args) {
                    args.emplace_back(to_llvm_type(a));
                }
                ret = to_llvm_type(func->ret);
                type_cache[ty] = llvm::FunctionType::get(ret, args, false);
                return type_cache.at(ty);
            }
            AKR_ASSERT(false);
        }
        type::Type process_type(const ast::AST &n) {
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
        type::StructType process_struct_decl(const ast::StructDecl &decl) {
            auto st = std::make_shared<type::StructTypeNode>();
            for (auto &field : decl->fields) {
                auto t = process_type(field->type);
                type::StructField f;
                f.index = (int)st->fields.size();
                f.name = field->var->identifier;
                f.type = t;
                st->fields.emplace_back(f);
            }
            st->name = decl->struct_name->name;
            types.emplace(st->name, st);
            return st;
        }
        void gen_prototype(const ast::FunctionDecl &decl) {
            auto ty = llvm::dyn_cast<llvm::FunctionType>(to_llvm_type(process_type(decl)));
            llvm::Function *F =
                llvm::Function::Create(ty, llvm::Function::ExternalLinkage, decl->name->identifier, owner.get());
            unsigned Idx = 0;
            for (auto &Arg : F->args())
                Arg.setName(decl->parameters[Idx++]->var->identifier);
            prototypes[decl->name->identifier] = F;
        }
        std::pair<ValueRecord, ValueRecord> arith_promote(const std::string &op, SourceLocation &loc,
                                                          const ValueRecord &lhs, const ValueRecord &rhs) {
            if (lhs.type->is_parent_of(rhs.type)) {
                return std::make_pair(lhs, cast(loc, rhs, lhs.type));
            } else if (rhs.type->is_parent_of(lhs.type)) {
                return std::make_pair(cast(loc, lhs, rhs.type), rhs);
            } else if (rhs.type->isa<type::PrimitiveType>() && lhs.type->isa<type::PrimitiveType>()) {
                return std::make_pair(cast(loc, lhs, type::float64), cast(loc, rhs, type::float64));
            } else {
                error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type->type_name(),
                                       rhs.type->type_name()));
            }
        }
        ValueRecord compile_expr(const ast::Expr &e) {
            if (e->isa<ast::Literal>()) {
                return compile_literal(e->cast<ast::Literal>());
            }
            if (e->isa<ast::Identifier>()) {
                return compile_var(e->cast<ast::Identifier>());
            }
            if (e->isa<ast::BinaryExpression>()) {
                auto binop = e->cast<ast::BinaryExpression>();
                auto op = binop->op;
                if (op == "||") {
                    auto *lhs_bb = builder->GetInsertBlock();
                    auto lhs = cast(binop->lhs->loc, compile_expr(binop->lhs), type::boolean);
                    auto *rhs_bb = llvm::BasicBlock::Create(ctx, "or_rhs", cur_function.function);
                    auto *merge_bb = llvm::BasicBlock::Create(ctx, "merge", cur_function.function);
                    builder->CreateCondBr(lhs.value, merge_bb, rhs_bb);
                    builder->SetInsertPoint(rhs_bb);
                    auto rhs = cast(binop->rhs->loc, compile_expr(binop->rhs), type::boolean);
                    builder->CreateBr(merge_bb);
                    builder->SetInsertPoint(merge_bb);
                    auto phi = builder->CreatePHI(to_llvm_type(type::boolean), 2, "or_phi");
                    phi->addIncoming(lhs.value, lhs_bb);
                    phi->addIncoming(rhs.value, rhs_bb);
                    return {phi, type::boolean};
                }
                if (op == "&&") {
                    auto *lhs_bb = builder->GetInsertBlock();
                    auto lhs = cast(binop->lhs->loc, compile_expr(binop->lhs), type::boolean);
                    auto *rhs_bb = llvm::BasicBlock::Create(ctx, "and_rhs", cur_function.function);
                    auto *merge_bb = llvm::BasicBlock::Create(ctx, "merge", cur_function.function);
                    builder->CreateCondBr(lhs.value, rhs_bb, merge_bb);
                    builder->SetInsertPoint(rhs_bb);
                    auto rhs = cast(binop->rhs->loc, compile_expr(binop->rhs), type::boolean);
                    builder->CreateBr(merge_bb);
                    builder->SetInsertPoint(merge_bb);
                    auto phi = builder->CreatePHI(to_llvm_type(type::boolean), 2, "and_phi");
                    phi->addIncoming(lhs.value, lhs_bb);
                    phi->addIncoming(rhs.value, rhs_bb);
                    return {phi, type::boolean};
                }
                auto lhs = compile_expr(binop->lhs);
                auto rhs = compile_expr(binop->rhs);
                auto [L, R] = arith_promote(op, binop->loc, lhs, rhs);
                AKR_ASSERT(L.type == R.type);
                if (op == "+") {
                    if (L.type->is_float()) {
                        return {builder->CreateFAdd(L.value, R.value), L.type};
                    } else {
                        return {builder->CreateAdd(L.value, R.value), L.type};
                    }
                } else if (op == "-") {
                    if (L.type->is_float()) {
                        return {builder->CreateFSub(L.value, R.value), L.type};
                    } else {
                        return {builder->CreateSub(L.value, R.value), L.type};
                    }
                } else if (op == "*") {
                    if (L.type->is_float()) {
                        return {builder->CreateFMul(L.value, R.value), L.type};
                    } else {

                        return {builder->CreateFMul(L.value, R.value), L.type};

                        return {builder->CreateMul(L.value, R.value), L.type};
                    }
                } else if (op == "/") {
                    if (L.type->is_float()) {
                        return {builder->CreateFDiv(L.value, R.value), L.type};
                    } else if (L.type->is_signed_int()) {
                        return {builder->CreateSDiv(L.value, R.value), L.type};
                    } else {
                        return {builder->CreateUDiv(L.value, R.value), L.type};
                    }
                } else if (op == "<") {
                    if (L.type->is_float()) {
                        return {builder->CreateFCmpOLT(L.value, R.value), type::boolean};
                    } else if (L.type->is_signed_int()) {
                        return {builder->CreateICmpSLT(L.value, R.value), type::boolean};
                    } else {
                        return {builder->CreateICmpULT(L.value, R.value), type::boolean};
                    }
                } else if (op == "<=") {
                    if (L.type->is_float()) {
                        return {builder->CreateFCmpOLE(L.value, R.value), type::boolean};
                    } else if (L.type->is_signed_int()) {
                        return {builder->CreateICmpSLE(L.value, R.value), type::boolean};
                    } else {
                        return {builder->CreateICmpULE(L.value, R.value), type::boolean};
                    }
                } else if (op == ">") {
                    if (L.type->is_float()) {
                        return {builder->CreateFCmpOGT(L.value, R.value), type::boolean};
                    } else if (L.type->is_signed_int()) {
                        return {builder->CreateICmpSGT(L.value, R.value), type::boolean};
                    } else {
                        return {builder->CreateICmpUGT(L.value, R.value), type::boolean};
                    }
                } else if (op == ">=") {
                    if (L.type->is_float()) {
                        return {builder->CreateFCmpOGE(L.value, R.value), type::boolean};
                    } else if (L.type->is_signed_int()) {
                        return {builder->CreateICmpSGE(L.value, R.value), type::boolean};
                    } else {
                        return {builder->CreateICmpUGE(L.value, R.value), type::boolean};
                    }
                } else if (op == "==") {
                    if (L.type->is_float()) {
                        return {builder->CreateFCmpOEQ(L.value, R.value), type::boolean};
                    } else {
                        return {builder->CreateICmpEQ(L.value, R.value), type::boolean};
                    }
                } else if (op == "!=") {
                    if (L.type->is_float()) {
                        return {builder->CreateFCmpONE(L.value, R.value), type::boolean};
                    } else {
                        return {builder->CreateICmpNE(L.value, R.value), type::boolean};
                    }
                }
            }
            AKR_ASSERT(false);
        }
        ValueRecord cast(const SourceLocation &loc, const ValueRecord &in, const type::Type &to) {
            // fmt::print("convert value from {} to {}\n", in.type->type_name(), to->type_name());
            if (in.type == to) {
                return in;
            }
            if (in.type->isa<type::PrimitiveType>() && to->isa<type::PrimitiveType>()) {
                if (to == type::boolean) {
                    if (in.type->is_int()) {
                        return {builder->CreateICmpEQ(in.value, llvm::ConstantInt::get(ctx, llvm::APInt(8, 0))), to};
                    } else {
                        return {builder->CreateFCmpOEQ(in.value, llvm::ConstantFP::get(ctx, llvm::APFloat(0.0))), to};
                    }
                }
                if (in.type->is_float()) {
                    if (to->is_int() && to->is_signed_int()) {
                        return {builder->CreateFPToSI(in.value, to_llvm_type(to)), to};
                    }
                    if (to->is_int() && !to->is_signed_int()) {
                        return {builder->CreateFPToUI(in.value, to_llvm_type(to)), to};
                    }
                    if (in.type == type::float32 && to == type::float64) {
                        return {builder->CreateFPExt(in.value, to_llvm_type(to)), to};
                    }
                    if (in.type == type::float64 && to == type::float32) {
                        return {builder->CreateFPTrunc(in.value, to_llvm_type(to)), to};
                    }
                    AKR_ASSERT(false);
                }
                if (in.type->is_int() && in.type->is_signed_int()) {
                    if (to->is_float()) {
                        return {builder->CreateSIToFP(in.value, to_llvm_type(to)), to};
                    }
                }
                if (in.type->is_int() && !in.type->is_signed_int()) {
                    if (to->is_float()) {
                        return {builder->CreateUIToFP(in.value, to_llvm_type(to)), to};
                    }
                }
                if (in.type->is_int() && to->is_int()) {
                    if (to->is_signed_int())
                        return {builder->CreateSExtOrTrunc(in.value, to_llvm_type(to)), to};
                    else
                        return {builder->CreateZExtOrTrunc(in.value, to_llvm_type(to)), to};
                }
            }
            if (in.type->isa<type::PrimitiveType>() && to->isa<type::VectorType>()) {
                auto v = to->cast<type::VectorType>();
                auto e = v->element_type;
                auto cvt = cast(loc, in, e);
                return {builder->CreateVectorSplat(v->count, cvt.value)};
            }
            if (in.type->isa<type::VectorType>() && to->isa<type::PrimitiveType>()) {
                error(loc, "cannot convert vector type to scalar");
            }
            error(loc, fmt::format("cannot convert value from {} to {}", in.type->type_name(), to->type_name()));
        }
        ValueRecord compile_literal(const ast::Literal &lit) {
            if (lit->isa<ast::FloatLiteral>()) {
                auto fl = lit->cast<ast::FloatLiteral>();
                return {llvm::ConstantFP::get(to_llvm_type(type::float32), fl->val), type::float32};
            }
            if (lit->isa<ast::IntLiteral>()) {
                auto i = lit->cast<ast::IntLiteral>();
                return {llvm::ConstantInt::get(to_llvm_type(type::int32), (int32_t)i->val), type::int32};
            }
            AKR_ASSERT(false);
        }
        void compile_ret(const ast::Return &ret) {
            auto r = compile_expr(ret->expr);
            builder->CreateRet(cast(ret->loc, r, cur_function.type->ret).value);
        }
        void assign_var(const SourceLocation &loc, const LValueRecord &lvalue, const ValueRecord &value) {
            auto cvt = cast(loc, value, lvalue.type);
            builder->CreateStore(cvt.value, lvalue.value);
        }
        LValueRecord eval_lvalue(const ast::Expr &e) {
            if (e->isa<ast::Identifier>()) {
                return vars.at(e->cast<ast::Identifier>()->identifier).value();
            }
            AKR_ASSERT(false);
        }
        ValueRecord compile_var(const ast::Identifier &var) {
            if (!vars.at(var->identifier).has_value()) {
                error(var->loc, fmt::format("identifier {} not found", var->identifier));
            }
            auto r = vars.at(var->identifier).value();
            return ValueRecord{builder->CreateLoad(r.value), r.type};
        }
        void compile_var_decl(const ast::VarDeclStmt &stmt) {
            auto decl = stmt->decl;
            auto ty = process_type(decl->type);
            llvm::IRBuilder<> tmp(&cur_function.function->getEntryBlock(),
                                  cur_function.function->getEntryBlock().begin());
            auto var = tmp.CreateAlloca(to_llvm_type(ty));
            vars.insert(decl->var->identifier, LValueRecord{var, ty});
            if (decl->init) {
                auto init = compile_expr(decl->init);
                assign_var(decl->var->loc, vars.at(decl->var->identifier).value(), init);
            }
        }
        void compile_assignment(const ast::Assignment &asgn) {
            auto lvalue = eval_lvalue(asgn->lhs);
            auto rvalue = compile_expr(asgn->rhs);
            assign_var(asgn->loc, lvalue, rvalue);
        }
        void compile_if(const ast::IfStmt &st) {
            auto cond = compile_expr(st->cond);
            cond = cast(st->loc, cond, type::boolean);
            if (st->if_false) {
                auto *then_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
                auto *else_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
                auto *merge_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);

                builder->CreateCondBr(cond.value, then_bb, else_bb);

                builder->SetInsertPoint(then_bb);
                compile_stmt(st->if_true);
                builder->CreateBr(merge_bb);

                builder->SetInsertPoint(else_bb);
                compile_stmt(st->if_false);
                builder->CreateBr(merge_bb);

                builder->SetInsertPoint(merge_bb);
            } else {
                auto *then_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
                auto *merge_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
                builder->CreateCondBr(cond.value, then_bb, merge_bb);
                builder->SetInsertPoint(then_bb);
                compile_stmt(st->if_true);
                builder->CreateBr(merge_bb);
                builder->SetInsertPoint(merge_bb);
            }
        }
        void compile_while(const ast::WhileStmt &st) {
            // auto prev_bb = builder->GetInsertBlock();
            auto cond_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            auto body_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            auto merge_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            builder->CreateBr(cond_bb);

            builder->SetInsertPoint(cond_bb);
            auto cond = compile_expr(st->cond);
            cond = cast(st->loc, cond, type::boolean);
            builder->CreateCondBr(cond.value, body_bb, merge_bb);

            builder->SetInsertPoint(body_bb);
            compile_stmt(st->body);
            builder->CreateBr(cond_bb);

            builder->SetInsertPoint(merge_bb);
        }
        void compile_stmt(const ast::Stmt &stmt) {
            if (stmt->isa<ast::VarDeclStmt>()) {
                compile_var_decl(stmt->cast<ast::VarDeclStmt>());
            } else if (stmt->isa<ast::Assignment>()) {
                compile_assignment(stmt->cast<ast::Assignment>());
            } else if (stmt->isa<ast::Return>()) {
                compile_ret(stmt->cast<ast::Return>());
            } else if (stmt->isa<ast::SeqStmt>()) {
                compile_block(stmt->cast<ast::SeqStmt>());
            } else if (stmt->isa<ast::IfStmt>()) {
                compile_if(stmt->cast<ast::IfStmt>());
            } else if (stmt->isa<ast::WhileStmt>()) {
                compile_while(stmt->cast<ast::WhileStmt>());
            } else {
                AKR_ASSERT(false);
            }
        }
        void compile_block(const ast::SeqStmt &stmt) {

            for (auto &s : stmt->stmts) {
                compile_stmt(s);
            }
        }
        void compile_func(const ast::FunctionDecl &func) {
            auto *F = prototypes.at(func->name->identifier);
            cur_function = FunctionRecord{F, process_type(func)->cast<type::FunctionType>()};
            for (uint32_t i = 0; i < func->parameters.size(); i++) {
                auto p = func->parameters[i];
                llvm::IRBuilder<> tmp(&cur_function.function->getEntryBlock(),
                                      cur_function.function->getEntryBlock().begin());
                auto *alloca = tmp.CreateAlloca(to_llvm_type(process_type(p->type)));

                vars.insert(p->var->identifier,
                            LValueRecord{tmp.CreateStore(F->getArg(i), alloca), process_type(p->type)});
            }
            auto *bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);

            builder = std::make_unique<llvm::IRBuilder<>>(bb);

            (void)compile_block(func->body);
            builder = nullptr;
        }

      public:
        LLVMBackend() {
            if (!init.lock()) {
                _init = std::make_shared<LLVMInit>();
                init = _init;
            } else {
                _init = init.lock();
            }
            init_types();
        }

        void compile(const Program &prog) override {
            owner = std::make_unique<llvm::Module>("test", ctx);
            for (auto &m : prog.modules) {
                for (auto &decl : m->structs) {
                    (void)process_struct_decl(decl);
                }
            }
            for (auto &m : prog.modules) {
                for (auto &decl : m->funcs) {
                    gen_prototype(decl);
                }
            }
            for (auto &m : prog.modules) {
                for (auto &decl : m->funcs) {
                    compile_func(decl);
                }
            }
            llvm::outs() << *owner << "\n";
            EE = llvm::EngineBuilder(std::move(owner)).create();
        }
        void *get_function_address(const std::string &name) { return (void *)EE->getFunctionAddress(name); }
        void add_function(const std::string &name, const type::FunctionType &ty, void *) {}
    };

    std::shared_ptr<Backend> create_llvm_backend() { return std::make_shared<LLVMBackend>(); }
} // namespace akari::asl