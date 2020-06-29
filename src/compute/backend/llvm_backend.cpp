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
        std::list<Module> todos;
        std::unordered_map<std::string, void *> modules;
        llvm::ExecutionEngine *EE = nullptr;
        void *do_compile(const Function &func) {return nullptr;}

      public:
        LLVMBackendImpl() {
            static_init();
            owner = std::make_unique<llvm::Module>("test", ctx);
        }
        Expected<void> add_module(const Module &m) override {
            todos.emplace_back(m);
            return {};
        }
        Expected<void> compile() override {
            for (auto &m : todos) {
                modules[m.name] = do_compile(m.function);
            }
            return {};
        }
        void *get_module_func(const std::string &name) override { return modules.at(name); }
        ~LLVMBackendImpl() {
            if (EE) {
                delete EE;
            }
        }
    };

    // std::shared_ptr<LLVMBackend> create_llvm_backend(){
    //     return std::make_shared<LLVMBackendImpl>();
    // }
} // namespace akari::compute