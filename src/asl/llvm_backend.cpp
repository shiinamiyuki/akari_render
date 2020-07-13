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
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
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

    struct Intrinsic {
        llvm::Intrinsic::ID id;
        int arity;
        std::function<bool(const type::Type &)> type_checker;
    };
    class Managler {
      public:
        std::string mangle(const type::Type & ty){
            if(ty->isa<type::PrimitiveType>()){
                return fmt::format("@P{}@p", ty->type_name());
            }else if(ty->isa<type::VectorType>()){
                auto v = ty->cast<type::VectorType>();
                return fmt::format("@V{}@{}n@v", mangle(v->element_type()), v->count);
            }else if(ty->isa<type::StructType>())
        }
        std::string mangle_arg(const type::Type & ty){
            return fmt::format("@A{}@a", mangle(ty));
        }
        std::string mangle(const std::string &name, const type::FunctionType &type) {

        }
    };
    class MCJITProgram : public Program {
        std::shared_ptr<LLVMInit> _init;
        std::unique_ptr<llvm::ExecutionEngine> EE;

      public:
        MCJITProgram(llvm::ExecutionEngine *EE) : EE(EE) {
            AKR_ASSERT(EE);
            _init = init.lock();
            AKR_ASSERT(_init);
        }
        void *get_function_pointer(const std::string &s) { return (void *)EE->getFunctionAddress(s); }
    };
    static thread_local llvm::LLVMContext ctx;
    class LLVMBackend : public Backend {

        std::unique_ptr<llvm::Module> owner;
        std::shared_ptr<LLVMInit> _init;
        FunctionRecord cur_function;
        std::unique_ptr<llvm::IRBuilder<>> builder;
        Environment<std::string, LValueRecord> vars;
        std::unordered_map<std::string, StructRecord> structs;
        std::unordered_map<std::string, type::Type> types;
        std::unordered_map<std::string, FunctionRecord> prototypes;
        std::unordered_map<type::Type, llvm::Type *> type_cache;
        std::unordered_map<std::string, Intrinsic> intrinsics;
        std::unique_ptr<llvm::legacy::FunctionPassManager> FPM;
        std::unique_ptr<llvm::legacy::PassManager> MPM;
        llvm::PassManagerBuilder pass_mgr_builder;
        llvm::BasicBlock *loop_pred = nullptr;
        llvm::BasicBlock *loop_merge = nullptr;
        bool is_terminated = false;
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
        void init_intrinsics() {
            std::function<bool(const type::Type &)> accept_fp_vector = [](const type::Type &ty) {
                if (ty->is_float() || ty->isa<type::VectorType>()) {
                    return true;
                }
                return false;
            };
            intrinsics.emplace("sin", Intrinsic{llvm::Intrinsic::sin, 1, accept_fp_vector});
            intrinsics.emplace("cos", Intrinsic{llvm::Intrinsic::cos, 1, accept_fp_vector});
            intrinsics.emplace("exp", Intrinsic{llvm::Intrinsic::exp, 1, accept_fp_vector});
            intrinsics.emplace("pow", Intrinsic{llvm::Intrinsic::pow, 2, accept_fp_vector});
            intrinsics.emplace("sqrt", Intrinsic{llvm::Intrinsic::sqrt, 1, accept_fp_vector});
            // intrinsics.emplace("reduce_min", Intrinsic{"reduce_min", accept_fp_vector});
            // intrinsics.emplace("reduce_max", Intrinsic{"reduce_max", accept_fp_vector});
            // intrinsics.emplace("reduce_add", Intrinsic{"reduce_add", accept_fp_vector});
            // intrinsics.emplace("reduce_mul", Intrinsic{"reduce_mul", accept_fp_vector});
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
                    if (a->isa<type::PrimitiveType>())
                        args.emplace_back(to_llvm_type(a));
                    else {
                        auto elem_type = to_llvm_type(a);
                        args.emplace_back(llvm::PointerType::get(elem_type, 0));
                    }
                }
                ret = to_llvm_type(func->ret);
                if (func->ret->isa<type::PrimitiveType>()) {
                    type_cache[ty] = llvm::FunctionType::get(ret, args, false);
                } else {
                    args.insert(args.begin(), llvm::PointerType::get(ret, 0));
                    type_cache[ty] = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), args, false);
                }
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
            structs[st->name] = {};
            return st;
        }
        void gen_prototype(const ast::FunctionDecl &decl) {
            auto f_ty = process_type(decl)->cast<type::FunctionType>();
            auto ty = llvm::dyn_cast<llvm::FunctionType>(to_llvm_type(f_ty));
            llvm::Function *F =
                llvm::Function::Create(ty, llvm::Function::ExternalLinkage, decl->name->identifier, owner.get());
            // unsigned Idx = 0;

            prototypes[decl->name->identifier] = FunctionRecord{F, f_ty};
        }
        std::pair<ValueRecord, ValueRecord> arith_promote(const std::string &op, const SourceLocation &loc,
                                                          const ValueRecord &lhs, const ValueRecord &rhs) {
            if (lhs.type->is_parent_of(rhs.type)) {
                return std::make_pair(lhs, cast(loc, rhs, lhs.type));
            } else if (rhs.type->is_parent_of(lhs.type)) {
                return std::make_pair(cast(loc, lhs, rhs.type), rhs);
            } else if (lhs.type->isa<type::VectorType>() && rhs.type->isa<type::PrimitiveType>()) {
                return std::make_pair(lhs, cast(loc, rhs, lhs.type));
            } else if (rhs.type->isa<type::VectorType>() && lhs.type->isa<type::PrimitiveType>()) {
                return std::make_pair(cast(loc, lhs, rhs.type), rhs);
            } else {
                error(loc, fmt::format("illegal binary op '{}' with {} and {}", op, lhs.type->type_name(),
                                       rhs.type->type_name()));
            }
        }
        ValueRecord binary_expr_helper(const SourceLocation &loc, const std::string &op, const ValueRecord &lhs,
                                       const ValueRecord &rhs) {
            auto [L, R] = arith_promote(op, loc, lhs, rhs);
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
            AKR_ASSERT(false);
        }
        ValueRecord compile_member_access(const ast::MemberAccess &access) {
            auto agg = compile_expr(access->var);
            if (agg.type->isa<type::VectorType>()) {
                auto v = agg.type->cast<type::VectorType>();
                auto member = access->member;
                auto n = v->count;
                int idx;
                if (member == "x" || member == "r") {
                    idx = 0;
                } else if (member == "y" || member == "g") {
                    idx = 1;
                } else if (member == "z" || member == "b") {
                    idx = 2;
                } else if (member == "w" || member == "a") {
                    idx = 3;
                }
                if (idx >= n) {
                    error(access->loc, "index out of bound");
                }
                return {builder->CreateExtractElement(agg.value, idx), v->element_type};
            } else if (agg.type->isa<type::StructType>()) {
                auto st = agg.type->cast<type::StructType>();
                for (size_t i = 0; i < st->fields.size(); i++) {
                    if (access->member == st->fields[i].name) {
                        return {builder->CreateExtractValue(agg.value, i), st->fields[i].type};
                    }
                }
                error(access->loc, fmt::format("{} cannot found in {}", access->member, get_typename_str(st)));
            }
            AKR_ASSERT(false);
        }
        void compile_assignment_member_access(const ast::Assignment &asgn) {
            auto access = asgn->lhs->cast<ast::MemberAccess>();
            AKR_ASSERT(access);
            auto var = eval_lvalue(access->var);
            if (var.type->isa<type::VectorType>()) {
                auto v = var.type->cast<type::VectorType>();
                auto member = access->member;
                auto n = v->count;
                int idx;
                if (member == "x" || member == "r") {
                    idx = 0;
                } else if (member == "y" || member == "g") {
                    idx = 1;
                } else if (member == "z" || member == "b") {
                    idx = 2;
                } else if (member == "w" || member == "a") {
                    idx = 3;
                }
                if (idx >= n) {
                    error(access->loc, "index out of bound");
                }
                auto val = cast(asgn->loc, compile_expr(asgn->rhs), v->element_type);
                auto tmp = builder->CreateLoad(var.value);
                auto new_vec = builder->CreateInsertElement(tmp, val.value, idx);
                builder->CreateStore(new_vec, var.value);
            } else if (var.type->isa<type::StructType>()) {
                auto st = var.type->cast<type::StructType>();
                for (size_t i = 0; i < st->fields.size(); i++) {
                    if (access->member == st->fields[i].name) {
                        auto val = cast(asgn->loc, compile_expr(asgn->rhs), st->fields[i].type);
                        auto gep = builder->CreateConstGEP2_32(to_llvm_type(var.type), var.value, 0, i);
                        builder->CreateStore(val.value, gep);
                        return;
                    }
                }
                error(access->loc, fmt::format("{} cannot found in {}", access->member, get_typename_str(st)));
            } else {
                AKR_ASSERT(false);
            }
        }

        ValueRecord compile_binary_expr(const ast::BinaryExpression &e) {
            auto binop = e->cast<ast::BinaryExpression>();
            auto op = binop->op;
            if (op == "||") {
                auto *lhs_bb = builder->GetInsertBlock();
                auto lhs = cast(binop->lhs->loc, compile_expr(binop->lhs), type::boolean);
                auto *rhs_bb = llvm::BasicBlock::Create(ctx, "or_rhs", cur_function.function);
                auto *merge_bb = llvm::BasicBlock::Create(ctx, "merge", cur_function.function);
                builder->CreateCondBr(lhs.value, merge_bb, rhs_bb);
                set_cur_bb(rhs_bb);
                auto rhs = cast(binop->rhs->loc, compile_expr(binop->rhs), type::boolean);
                builder->CreateBr(merge_bb);
                set_cur_bb(merge_bb);
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
                set_cur_bb(rhs_bb);
                auto rhs = cast(binop->rhs->loc, compile_expr(binop->rhs), type::boolean);
                builder->CreateBr(merge_bb);
                set_cur_bb(merge_bb);
                auto phi = builder->CreatePHI(to_llvm_type(type::boolean), 2, "and_phi");
                phi->addIncoming(lhs.value, lhs_bb);
                phi->addIncoming(rhs.value, rhs_bb);
                return {phi, type::boolean};
            }
            auto lhs = compile_expr(binop->lhs);
            auto rhs = compile_expr(binop->rhs);
            return binary_expr_helper(binop->loc, op, lhs, rhs);
        }
        ValueRecord default_initialize(const SourceLocation &loc, const type::Type &ty) {
            if (ty->isa<type::PrimitiveType>() && ty->is_int()) {
                return {llvm::ConstantInt::get(to_llvm_type(ty), 0), ty};
            }
            if (ty->isa<type::PrimitiveType>() && ty->is_float()) {
                return {llvm::ConstantFP::get(to_llvm_type(ty), 0), ty};
            }
            if (ty->isa<type::VectorType>()) {
                auto v = ty->cast<type::VectorType>();
                return {builder->CreateVectorSplat(v->count, default_initialize(loc, v->element_type).value), v};
            }
            AKR_ASSERT(false);
        }
        ValueRecord compile_ctor_call(const ast::ConstructorCall &call) {
            auto type = process_type(call->type);
            if (type->isa<type::PrimitiveType>()) {
                if (call->args.size() != 1) {
                    error(call->loc, fmt::format("call to `constructor {}` expected 1 argumnet but found {}",
                                                 get_typename_str(type), call->args.size()));
                }
                auto a = compile_expr(call->args[0]);
                return cast(call->loc, a, type);
            } else if (type->isa<type::VectorType>()) {
                auto vt = type->cast<type::VectorType>();
                // auto cnt = vt->count;
                auto n_args = call->args.size();
                // if (n_args != 1 && n_args != cnt) {
                //     error(call->loc, fmt::format("call to `constructor {}` expected 1 or {} argumnets but found {}",
                //                                  get_typename_str(type), cnt, n_args));
                // }
                if (n_args == 1) {
                    auto a = compile_expr(call->args[0]);
                    return cast(call->loc, a, type);

                } else {
                    auto vec =
                        builder->CreateVectorSplat(vt->count, default_initialize(call->loc, vt->element_type).value);
                    int emitted_cnt = 0;
                    for (int i = 0; i < n_args; i++) {
                        auto a = compile_expr(call->args[i]);
                        if (auto u = a.type->cast<type::VectorType>()) {
                            if (u->element_type != vt->element_type) {
                                error(call->args[i]->loc,
                                      fmt::format(
                                          "in call to `constructor {}` argument {}, implicit conversion is illegal",
                                          get_typename_str(vt), i + 1));
                            }
                            if (emitted_cnt + u->count > vt->count) {
                                error(call->args[i]->loc,
                                      fmt::format("in call to `constructor {}` {} elements provided, only expects {}",
                                                  get_typename_str(vt), emitted_cnt + u->count, vt->count));
                            }
                            for (int j = 0; j < u->count; j++) {
                                auto elem = builder->CreateExtractElement(a.value, j);
                                vec = builder->CreateInsertElement(vec, elem, emitted_cnt + j);
                            }
                            emitted_cnt += u->count;
                        } else if (auto _ = a.type->cast<type::PrimitiveType>()) {
                            auto elem = cast(call->args[i]->loc, a, vt->element_type);
                            vec = builder->CreateInsertElement(vec, elem.value, emitted_cnt);
                            emitted_cnt += 1;
                        }
                    }
                    if (emitted_cnt < vt->count) {
                        auto elem = default_initialize(call->loc, vt->element_type);
                        for (int k = emitted_cnt; k < vt->count; k++) {
                            vec = builder->CreateInsertElement(vec, elem.value, k);
                        }
                    }
                    return {vec, vt};
                }
            } else {
            }
            AKR_ASSERT(false);
        }
        ValueRecord compile_intrinsic_call(const ast::FunctionCall &call) {
            auto &intrinsic = intrinsics.at(call->func->identifier);
            auto arity = intrinsic.arity;
            if (arity != call->args.size()) {
                error(call->loc, fmt::format("call to intrinsic `{}` {} arguments expected but found {}",
                                             call->func->identifier, arity, call->args.size()));
            }
            std::vector<llvm::Value *> llvm_args;
            type::Type arg_type = nullptr;
            for (auto &a : call->args) {
                auto arg = compile_expr(a);
                if (!intrinsic.type_checker(arg.type)) {
                    error(call->loc,
                          fmt::format("call to intrinsic `{}` type {} of argument {} is not support",
                                      call->func->identifier, get_typename_str(arg.type), llvm_args.size() + 1));
                }
                if (!arg_type) {
                    arg_type = arg.type;
                } else {
                    arg = cast(a->loc, arg, arg_type);
                }
                llvm_args.emplace_back(arg.value);
            }
            if (arity == 1) {
                return {builder->CreateUnaryIntrinsic(intrinsic.id, llvm_args.at(0)), arg_type};
            } else if (arity == 2) {
                return {builder->CreateBinaryIntrinsic(intrinsic.id, llvm_args.at(0), llvm_args.at(1)), arg_type};
            } else {
                AKR_ASSERT(false);
            }
        }
        ValueRecord compile_func_call(const ast::FunctionCall &call) {
            auto func = call->func->identifier;
            if (intrinsics.count(func)) {
                return compile_intrinsic_call(call);
            }
            if (!prototypes.count(func)) {
                error(call->loc, fmt::format("{} is not a function or it is undefined", func));
            }
            auto &f_rec = prototypes.at(func);
            std::vector<ValueRecord> args;
            auto expected_num = f_rec.type->args.size();
            auto actual_num = call->args.size();
            if (expected_num != actual_num) {
                error(call->loc, fmt::format("{} arguments expected but found {}", expected_num, actual_num));
            }
            for (int i = 0; i < f_rec.type->args.size(); i++) {
                auto &expected_ty = f_rec.type->args[i];
                auto arg = compile_expr(call->args[i]);
                // auto &arg_ty = arg.type;
                args.emplace_back(cast(call->args[i]->loc, arg, expected_ty));
            }
            std::vector<llvm::Value *> llvm_args;
            for (auto &a : args) {
                if (a.type->isa<type::PrimitiveType>())
                    llvm_args.emplace_back(a.value);
                else {
                    // pass byval
                    auto copy = create_entry_block_alloca(a.type);
                    builder->CreateStore(a.value, copy);
                    llvm_args.emplace_back(copy);
                }
            }
            if (f_rec.type->ret->isa<type::PrimitiveType>()) {
                return {builder->CreateCall(f_rec.function, llvm_args), f_rec.type->ret};
            } else {
                auto ret_value = create_entry_block_alloca(f_rec.type->ret);
                llvm_args.insert(llvm_args.begin(), ret_value);
                builder->CreateCall(f_rec.function, llvm_args);
                return {builder->CreateLoad(ret_value), f_rec.type->ret};
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
        static std::string get_typename_str(const type::Type &ty) {
            if (ty == type::float32) {
                return "float";
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
            if (ty->isa<type::VectorType>()) {
                auto v = ty->cast<type::VectorType>();
                return fmt::format("vec[{} x {}]", v->count, get_typename_str(v->element_type));
            }
            if (ty->isa<type::StructType>()) {
                return fmt::format("struct {}", ty->cast<type::StructType>()->name);
            }
            return "unknown";
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
                        error(loc, fmt::format("implicit conversion from {} to {} is not allowed",
                                               get_typename_str(in.type), get_typename_str(to)));
                    }
                }
                if (in.type->is_float()) {
                    if (to->is_int()) {
                        error(loc, fmt::format("implicit conversion from {} to {} is not allowed",
                                               get_typename_str(in.type), get_typename_str(to)));
                    }
                    if (in.type == type::float32 && to == type::float64) {
                        return {builder->CreateFPExt(in.value, to_llvm_type(to)), to};
                    }
                    if (in.type == type::float64 && to == type::float32) {
                        return {builder->CreateFPTrunc(in.value, to_llvm_type(to)), to};
                    }
                    AKR_ASSERT(false);
                }
                if (in.type->is_int() && to->is_float()) {
                    if (to->is_int()) {
                        error(loc, fmt::format("implicit conversion from {} to {} is not allowed",
                                               get_typename_str(in.type), get_typename_str(to)));
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
                return {builder->CreateVectorSplat(v->count, cvt.value), to};
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
            // auto *term_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            if (cur_function.type->ret->isa<type::PrimitiveType>())
                builder->CreateRet(cast(ret->loc, r, cur_function.type->ret).value);
            else {
                builder->CreateStore(r.value, cur_function.function->getArg(0));
                builder->CreateRetVoid();
            }
            is_terminated = true;
        }
        void assign_var(const SourceLocation &loc, const LValueRecord &lvalue, const ValueRecord &value) {
            auto cvt = cast(loc, value, lvalue.type);
            builder->CreateStore(cvt.value, lvalue.value);
        }
        LValueRecord eval_lvalue(const ast::Expr &e) {
            if (e->isa<ast::Identifier>()) {
                return vars.at(e->cast<ast::Identifier>()->identifier).value();
            }
            // else if (e->isa<ast::MemberAccess>()) {
            //     return eval_lvalue_member_access(e->cast<ast::MemberAccess>());
            // }
            AKR_ASSERT(false);
        }
        ValueRecord compile_var(const ast::Identifier &var) {
            if (!vars.at(var->identifier).has_value()) {
                error(var->loc, fmt::format("identifier {} not found", var->identifier));
            }
            auto r = vars.at(var->identifier).value();
            return ValueRecord{builder->CreateLoad(r.value), r.type};
        }
        void compile_var_decl(const ast::VarDecl &decl) {
            auto ty = process_type(decl->type);

            auto var = create_entry_block_alloca(ty);
            vars.insert(decl->var->identifier, LValueRecord{var, ty});
            if (decl->init) {
                auto init = compile_expr(decl->init);
                assign_var(decl->var->loc, vars.at(decl->var->identifier).value(), init);
            } else {
                auto init = default_initialize(decl->loc, ty);
                assign_var(decl->var->loc, vars.at(decl->var->identifier).value(), init);
            }
        }
        void compile_var_decl(const ast::VarDeclStmt &stmt) {
            auto decl = stmt->decl;
            return compile_var_decl(decl);
        }

        void compile_assignment(const ast::Assignment &asgn) {
            if (asgn->lhs->isa<ast::MemberAccess>()) {
                compile_assignment_member_access(asgn);
            } else {
                auto lvalue = eval_lvalue(asgn->lhs);
                auto rvalue = compile_expr(asgn->rhs);
                assign_var(asgn->loc, lvalue, rvalue);
            }
        }
        void set_cur_bb(llvm::BasicBlock *bb) {
            is_terminated = false;
            builder->SetInsertPoint(bb);
        }
        void compile_if(const ast::IfStmt &st) {
            auto cond = compile_expr(st->cond);
            cond = cast(st->loc, cond, type::boolean);
            if (st->if_false) {
                auto *then_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
                auto *else_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
                auto *merge_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);

                builder->CreateCondBr(cond.value, then_bb, else_bb);

                set_cur_bb(then_bb);
                compile_stmt(st->if_true);
                if (!is_terminated)
                    builder->CreateBr(merge_bb);

                set_cur_bb(else_bb);
                compile_stmt(st->if_false);
                if (!is_terminated)
                    builder->CreateBr(merge_bb);

                set_cur_bb(merge_bb);
            } else {
                auto *then_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
                auto *merge_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
                builder->CreateCondBr(cond.value, then_bb, merge_bb);
                set_cur_bb(then_bb);
                compile_stmt(st->if_true);
                if (!is_terminated)
                    builder->CreateBr(merge_bb);
                set_cur_bb(merge_bb);
            }
        }
        void compile_for(const ast::ForStmt &st) {
            // auto prev_bb = builder->GetInsertBlock();
            auto _ = vars.push();
            if (st->init)
                compile_var_decl(st->init);
            auto cond_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            auto body_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            auto merge_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);

            auto tmp_pre = loop_pred;
            auto tmp_merge = loop_merge;

            loop_pred = cond_bb;
            loop_merge = merge_bb;

            builder->CreateBr(cond_bb);

            set_cur_bb(cond_bb);
            auto cond = compile_expr(st->cond);
            cond = cast(st->loc, cond, type::boolean);
            builder->CreateCondBr(cond.value, body_bb, merge_bb);

            set_cur_bb(body_bb);
            compile_stmt(st->body);
            if (!is_terminated) {
                compile_stmt(st->step);
                builder->CreateBr(cond_bb);
            }

            set_cur_bb(merge_bb);

            loop_pred = tmp_pre;
            loop_merge = tmp_merge;
        }
        void compile_while(const ast::WhileStmt &st) {
            // auto prev_bb = builder->GetInsertBlock();
            auto cond_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            auto body_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            auto merge_bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);

            auto tmp_pre = loop_pred;
            auto tmp_merge = loop_merge;

            loop_pred = cond_bb;
            loop_merge = merge_bb;

            builder->CreateBr(cond_bb);

            set_cur_bb(cond_bb);
            auto cond = compile_expr(st->cond);
            cond = cast(st->loc, cond, type::boolean);
            builder->CreateCondBr(cond.value, body_bb, merge_bb);

            set_cur_bb(body_bb);
            compile_stmt(st->body);
            if (!is_terminated)
                builder->CreateBr(cond_bb);

            set_cur_bb(merge_bb);

            loop_pred = tmp_pre;
            loop_merge = tmp_merge;
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
            } else if (stmt->isa<ast::ForStmt>()) {
                compile_for(stmt->cast<ast::ForStmt>());
            }else if (stmt->isa<ast::BreakStmt>()) {
                auto st = (stmt->cast<ast::BreakStmt>());
                if (!loop_pred) {
                    error(stmt->loc, "`break` outside of loop!");
                }
                builder->CreateBr(loop_merge);
            } else if (stmt->isa<ast::BreakStmt>()) {
                auto st = (stmt->cast<ast::BreakStmt>());
                if (!loop_pred) {
                    error(stmt->loc, "`continue` outside of loop!");
                }
                builder->CreateBr(loop_pred);
            } else {
                AKR_ASSERT(false);
            }
        }
        void compile_block(const ast::SeqStmt &stmt) {

            for (auto &s : stmt->stmts) {
                compile_stmt(s);
            }
        }
        llvm::AllocaInst *create_entry_block_alloca(const type::Type &ty) {
            llvm::IRBuilder<> tmp(&cur_function.function->getEntryBlock(),
                                  cur_function.function->getEntryBlock().begin());
            return tmp.CreateAlloca(to_llvm_type(ty));
        }
        void compile_func(const ast::FunctionDecl &func) {
            auto *F = prototypes.at(func->name->identifier).function;
            cur_function = FunctionRecord{F, process_type(func)->cast<type::FunctionType>()};
            auto *bb = llvm::BasicBlock::Create(ctx, "", cur_function.function);
            auto _ = vars.push();
            builder = std::make_unique<llvm::IRBuilder<>>(bb);
            int f_arg_offset = 0;
            if (!cur_function.type->ret->isa<type::PrimitiveType>()) {
                f_arg_offset = 1;
                // F->getArg(0)->addAttr(llvm::Attribute::StructRet);
                // F->getArg(0)->addAttr(llvm::Attribute::NoAlias);
            }
            // fmt::print("compiling {}\n", func->name->identifier);
            for (uint32_t i = 0; i < func->parameters.size(); i++) {
                auto p = func->parameters[i];
                auto p_ty = process_type(p->type);
                if (p_ty->isa<type::PrimitiveType>()) {
                    auto *alloca = create_entry_block_alloca(p_ty);
                    builder->CreateStore(F->getArg(i + f_arg_offset), alloca);
                    vars.insert(p->var->identifier, LValueRecord{alloca, p_ty});
                    // fmt::print("store prim args\n");
                } else {
                    // F->getArg(i + f_arg_offset)->addAttr(llvm::Attribute::NoAlias);
                    // F->getArg(i + f_arg_offset)->addAttr(llvm::Attribute::ByVal);
                    vars.insert(p->var->identifier, LValueRecord{F->getArg(i + f_arg_offset), p_ty});
                }
                fflush(stdout);
                F->getArg(i + f_arg_offset)->setName(func->parameters[i]->var->identifier);
            }

            compile_block(func->body);
            llvm::verifyFunction(*F);
            if (opt.opt_level != CompileOptions::OptLevel::OFF) {
                FPM->run(*F);
            }
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
            init_intrinsics();
        }
        CompileOptions opt;
        std::shared_ptr<Program> compile(const ParsedProgram &prog, const CompileOptions &_opt) override {
            opt = _opt;
            owner = std::make_unique<llvm::Module>("test", ctx);
            pass_mgr_builder.OptLevel = 2;
            FPM = std::make_unique<llvm::legacy::FunctionPassManager>(owner.get());
            MPM = std::make_unique<llvm::legacy::PassManager>();
            pass_mgr_builder.populateFunctionPassManager(*FPM);
            pass_mgr_builder.populateModulePassManager(*MPM);
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
            if (opt.opt_level != CompileOptions::OptLevel::OFF) {
                MPM->run(*owner);
            }
            llvm::outs() << *owner << "\n";
            auto EE = llvm::EngineBuilder(std::move(owner)).create();
            AKR_ASSERT(EE->getFunctionAddress("main") != 0);
            return std::make_shared<MCJITProgram>(EE);
        }

        void add_function(const std::string &name, const type::FunctionType &ty, void *) {}
    };

    std::shared_ptr<Backend> create_llvm_backend() { return std::make_shared<LLVMBackend>(); }
} // namespace akari::asl