
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
namespace akari::asl {
    CodeGenerator::CodeGenerator() { add_predefined_types(); }
    static type::Type create_vec_type(const type::Type &base, int n) {
        auto v = std::make_shared<type::VectorTypeNode>();
        v->element_type = base;
        v->count = n;
        return v;
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
        for (auto &m : program.translation_units) {
            for (auto &decl : m->structs) {
                (void)process_struct_decl(decl);
            }
        }
    }
    void CodeGenerator::process_prototypes() {
        for (auto &m : program.translation_units) {
            for (auto &decl : m->funcs) {
                auto f_ty = process_type(decl)->cast<type::FunctionType>();
                prototypes[decl->name->identifier].overloads.emplace_back(f_ty);
            }
        }
    }
    class CodeGenCPP : public CodeGenerator {
      public:
        virtual std::string do_generate() { return ""; }
    };
    std::unique_ptr<CodeGenerator> cpp_generator() { return std::make_unique<CodeGenCPP>(); }
} // namespace akari::asl