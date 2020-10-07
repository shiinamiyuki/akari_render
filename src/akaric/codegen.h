
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
#include <sstream>
#include <akaric/ast.h>
#include <akaric/parser.h>
namespace akari::asl {
    struct BuildConfig {};
    struct Module {
        std::string name;
        std::vector<std::string> type_parameters;
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
            } else if (ty->isa<type::OpaqueType>()) {
                return fmt::format("ZO{}Zo", ty->cast<type::OpaqueType>()->name);
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
    class Twine {
        struct Node {
            std::string s;
            std::shared_ptr<Node> prev, next;
            Node() = default;
            Node(const std::string &s) : s(s) {}
            void str(std::ostringstream &os) {
                if (prev) {
                    prev->str(os);
                }
                os << s;
                if (next) {
                    next->str(os);
                }
            }
        };
        std::shared_ptr<Node> head;

      public:
        Twine() : Twine(std::string("")) {}
        Twine(const char *s) : head(std::make_shared<Node>(s)) {}
        Twine(const std::string &s) : head(std::make_shared<Node>(s)) {}
        Twine &append(const Twine &s) {
            if (!head->next) {
                head->next = s.head;
            } else {
                auto node = std::make_shared<Node>();
                node->prev = head;
                node->next = s.head;
                head = node;
            }
            return *this;
        }
        static Twine concat(const Twine &a, const Twine &b) {
            Twine cat = a;
            cat.append(b);
            return cat;
        }
        static Twine concat(const Twine &a, const std::string &s, const Twine &b) {
            Twine cat;
            auto node = std::make_shared<Node>(s);
            node->prev = a.head;
            node->next = b.head;
            cat.head = node;
            return cat;
        }
        [[nodiscard]] std::string str() const {
            std::ostringstream os;
            // std::vector<std::shared_ptr<Node>> st;
            // auto node = head;
            // while (!st.empty() && node) {
            //     if (node) {
            //         st.push_back(node);
            //         node = node->prev;
            //     } else {
            //         node = st.back();
            //         st.pop_back();
            //         os << node->s;
            //         node = node->next;
            //     }
            // }
            head->str(os);
            return os.str();
        }
    };
    class CodeGenerator {
      protected:
        struct FunctionRecord {
            std::unordered_map<std::string, type::FunctionType> overloads;
        };
        struct ValueRecord {
            Twine value;
            type::AnnotatedType annotated_type;
            ValueRecord() = default;
            type::Type type() const { return annotated_type.type; }
        };
        Environment<std::string, ValueRecord> vars;
        std::unordered_map<std::string, type::StructType> structs;
        std::unordered_map<std::string, type::Type> types;
        std::unordered_map<std::string, FunctionRecord> prototypes;
        type::AnnotatedType process_type(const ast::AST &n);
        type::StructType process_struct_decl(const ast::StructDecl &decl);
        void process_buffer_decls();
        void process_uniform_decls();
        void process_struct_decls();
        void process_prototypes();
        void add_predefined_types();
        Module module;
        BuildConfig config;
        virtual std::string do_generate() = 0;
        int indent = 0;
        template <class... Args>
        void wl(std::ostringstream &os, const std::string &s, Args &&... args) {
            for (int i = 0; i < indent; i++) {
                os << "    ";
            }
            os << fmt::format(s, std::forward<Args>(args)...) << "\n";
        }
        template <class F>
        void with_block(F &&f) {
            indent++;
            f();
            indent--;
        }
        [[noreturn]] void error(const SourceLocation &loc, std::string &&msg) {
            throw std::runtime_error(fmt::format("error: {} at {}:{}:{}", msg, loc.filename, loc.line, loc.col));
        }
        void warning(const SourceLocation &loc, std::string &&msg) {
            fmt::print("warning: {} at {}:{}:{}", msg, loc.filename, loc.line, loc.col);
        }
        void add_type_parameters();

      public:
        CodeGenerator();
        std::string generate(const BuildConfig &config_, const Module &module_) {
            this->config = config_;
            this->module = module_;
            add_type_parameters();
            process_struct_decls();
            process_prototypes();
            process_buffer_decls();
            process_uniform_decls();
            return do_generate();
        }
    };

    std::unique_ptr<CodeGenerator> cpp_generator();
    std::unique_ptr<CodeGenerator> cuda_generator();
} // namespace akari::asl
