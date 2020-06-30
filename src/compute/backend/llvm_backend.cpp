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
#include <akari/core/akari.h>
#include <akari/core/platform.h>

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

#include <akari/compute/backend.h>
#include <mutex>
#include <iostream>

namespace akari::compute {
    using namespace ir;
    struct LLVMInit {
        LLVMInit() {
            llvm::InitializeNativeTarget();
            LLVMInitializeNativeAsmPrinter();
            LLVMInitializeNativeAsmParser();
        }
        ~LLVMInit() { llvm::llvm_shutdown(); }
    };
    static void static_init() {
        static std::once_flag flag;
        static std::unique_ptr<LLVMInit> p;
        std::call_once(flag, [&]() { p = std::make_unique<LLVMInit>(); });
    }
    class LLVMBackendImpl : public LLVMBackend {
        llvm::LLVMContext ctx;
        std::unique_ptr<llvm::Module> owner;

        std::unordered_map<ir::Function, void *> cache;
        llvm::ExecutionEngine *EE = nullptr;
        llvm::Function *cur_function = nullptr;
        std::unique_ptr<llvm::IRBuilder<>> builder;
        struct EnvironmentFrame {
            std::shared_ptr<EnvironmentFrame> parent;
            std::unordered_map<ir::Var, llvm::Value *> map;
            void extend(const ir::Var &v, llvm::Value *e) { map.emplace(v, e); }
            llvm::Value *lookup(const ir::Var &v) const {
                if (map.count(v)) {
                    return map.at(v);
                }
                if (parent) {
                    return parent->lookup(v);
                } else {
                    std::cerr << "error looking up %" << v->name() << std::endl;
                    std::abort();
                }
            }
        };

        llvm::Type *to_llvm_type(const ir::Type &t) {
            if (t == get_primitive_type(ir::PrimitiveTy::float32)) {
                return llvm::Type::getFloatTy(ctx);
            }
            if (t == get_primitive_type(ir::PrimitiveTy::float64)) {
                return llvm::Type::getDoubleTy(ctx);
            }
            return nullptr;
        }

        llvm::Value *compile_expr(const ir::Expr &expr, const std::shared_ptr<EnvironmentFrame> &env) {
            if (expr->isa<Constant>()) {
                auto cst = expr->cast<Constant>();
                return std::visit(
                    [&](auto &&arg) -> llvm::Value * {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<float, T>) {
                            return llvm::ConstantFP::get(to_llvm_type(cst->type), arg);
                        } else if constexpr (std::is_same_v<double, T>) {
                            return llvm::ConstantFP::get(to_llvm_type(cst->type), arg);
                        } else if constexpr (std::is_same_v<int32_t, T>) {
                            return llvm::ConstantInt::get(to_llvm_type(cst->type), arg, true);
                        } else {
                            static_assert(false);
                        }
                    },
                    cst->value());
            } else if (expr->isa<Call>()) {
                auto call = expr->cast<Call>();
                auto args = call->args();
                auto op = call->op();
                std::vector<llvm::Value *> values;
                for (auto &a : args) {
                    values.emplace_back(compile_expr(a, env));
                }
                if (op->isa<Primitive>()) {
                    auto prim = op->cast<Primitive>()->primitive();
                    if (prim == PrimitiveOp::FAdd) {
                        return builder->CreateFAdd(values.at(0),values.at(1));
                    }else if (prim == PrimitiveOp::FSub) {
                        return builder->CreateFSub(values.at(0),values.at(1));
                    }else if (prim == PrimitiveOp::FMul) {
                        return builder->CreateFMul(values.at(0),values.at(1));
                    }else if (prim == PrimitiveOp::FDiv) {
                        return builder->CreateFDiv(values.at(0),values.at(1));
                    }else{
                        std::abort();
                    }
                }
            } else if(expr->isa<Var>()){
                return env->lookup(expr->cast<Var>());
            } else if(expr->isa<Let>()){
                auto let = expr->cast<Let>();
                auto val = compile_expr(let->value(), env);
                auto new_env = std::make_shared<EnvironmentFrame>();
                new_env->parent = env;
                new_env->extend(let->var(), val);
                return compile_expr(let->body(), new_env);
            }
            std::abort();
        }

        void *do_compile(const Function &func) {
            owner = std::make_unique<llvm::Module>("test", ctx);
            std::vector<llvm::Type *> args_t;
            llvm::Type *ret_t;
            for (auto &p : func->parameters()) {
                args_t.emplace_back(to_llvm_type(p->type));
            }
            ret_t = to_llvm_type(func->body()->type);
            cur_function = llvm::Function::Create(llvm::FunctionType::get(ret_t, args_t, false),
                                                  llvm::Function::ExternalLinkage, "default", owner.get());

            auto *bb = llvm::BasicBlock::Create(ctx, "", cur_function);
            builder = std::make_unique<llvm::IRBuilder<>>(bb);

            auto env = std::make_shared<EnvironmentFrame>();
            int cnt = 0;
            for(auto it = cur_function->arg_begin(); it != cur_function->arg_end(); it++){
                auto * arg = &*it;
                env->extend(func->parameters()[cnt], arg);
                cnt++;
            }
            builder->CreateRet(compile_expr(func->body(), env));
            llvm::outs() << *owner << "\n";
            EE = llvm::EngineBuilder(std::move(owner)).create();
            return (void*)EE->getFunctionAddress("default");
        }

      public:
        LLVMBackendImpl() {
            static_init();
            
        }
        void *compile(const ir::Function &func) {
            if (cache.count(func)) {
                return cache.at(func);
            }
            auto fp = do_compile(func);
            cache[func] = fp;
            return fp;
        }
        ~LLVMBackendImpl() {
            if (EE) {
                delete EE;
            }
        }
    };
    static std::shared_ptr<LLVMBackend> llvm_backend;
    std::shared_ptr<LLVMBackend> create_llvm_backend(){
        if(!llvm_backend){
            llvm_backend = std::make_shared<LLVMBackendImpl>();
        }
        return llvm_backend;
    }
} // namespace akari::compute