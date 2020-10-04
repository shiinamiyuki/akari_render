
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
#include <akaric/ast.h>
#include <akaric/parser.h>
namespace akari::asl {
    struct BuildConfig {};
    struct ParsedProgram {
        std::vector<ast::TopLevel> translation_units;
    };
    class Mangler {
        std::string mangle(const type::Type &ty) {
            if (ty->isa<type::PrimitiveType>()) {
                return fmt::format("ZP{}Zp", ty->type_name());
            } else if (ty->isa<type::VectorType>()) {
                auto v = ty->cast<type::VectorType>();
                return fmt::format("ZV{}Z{}nZv", mangle(v->element_type), v->count);
            } else if (ty->isa<type::StructType>()) {
                return fmt::format("ZS{}Zs", ty->cast<type::StructType>()->name);
            } else {
                AKR_ASSERT(false);
            }
        }
        std::string mangle_arg(const type::Type &ty) { return fmt::format("ZA{}Za", mangle(ty)); }

      public:
        std::string mangle(const std::string &name, const std::vector<type::Type> &args) {
            std::string s = fmt::format("ZF{}", name);
            for (auto &a : args) {
                s.append(mangle_arg(a));
            }
            return s.append("Zf");
        }
    };
    class CodeGenerator {
      protected:
        struct FunctionRecord {
            std::vector<type::FunctionType> overloads;
        };
        Environment<std::string, type::Type> vars;
        std::unordered_map<std::string, type::StructType> structs;
        std::unordered_map<std::string, type::Type> types;
        std::unordered_map<std::string, FunctionRecord> prototypes;
        type::Type process_type(const ast::AST &n);
        type::StructType process_struct_decl(const ast::StructDecl &decl);
        void process_struct_decls();
        void process_prototypes();
        void add_predefined_types();
        ParsedProgram program;
        BuildConfig config;
        virtual std::string do_generate() = 0;
        
      public:
        CodeGenerator();
        std::string generate(const BuildConfig &config_, const ParsedProgram &program_) {
            this->config = config_;
            this->program = program_;
            process_struct_decls();
            process_prototypes();
            return do_generate();
        }
    };

    std::unique_ptr<CodeGenerator> cpp_generator();
} // namespace akari::asl
