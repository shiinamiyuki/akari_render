
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
        std::vector<ast::TopLevel> translation_units;
    };

    class Mangler {
        std::string mangle(const type::Type &ty) {
            if (ty == type::float32) {
                return "f";
            } else if (ty == type::float64) {
                return "d";
            } else if (ty == type::int32) {
                return "i";
            } else if (ty == type::uint32) {
                return "u";
            } else if (ty == type::boolean) {
                return "b";
            } else if (ty->isa<type::VectorType>()) {
                auto v = ty->cast<type::VectorType>();
                return fmt::format("ZV{}Z{}nZv", mangle(v->element_type), v->count);
            } else if (ty->isa<type::MatrixType>()) {
                auto m = ty->cast<type::MatrixType>();
                return fmt::format("ZM{}Zc{}Zr{}Zm", mangle(m->element_type), m->cols, m->rows);
            } else if (ty->isa<type::StructType>()) {
                return fmt::format("ZS{}Zs", ty->cast<type::StructType>()->name);
            } else if (ty->isa<type::OpaqueType>()) {
                return fmt::format("ZO{}Zo", ty->cast<type::OpaqueType>()->name);
            } else if (auto q = ty->cast<type::QualifiedType>()) {
                return mangle(q->element_type);
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
            struct Entry {
                type::FunctionType type;
                bool is_intrinsic = false;
                Entry(type::FunctionType type, bool is_intrinsic) : type(type), is_intrinsic(is_intrinsic) {}
            };
            std::unordered_map<std::string, Entry> overloads;
        };
        struct ValueRecord {
            Twine value;
            type::Type type_;
            ValueRecord() = default;
            type::Type type() const { return type_; }
        };
        Environment<std::string, ValueRecord> vars;
        Environment<std::string, int> const_ints;
        struct CodeBlock {
            std::vector<std::pair<int, std::string>> lines;
            int indent = 0;
            template <class... Args>
            void wl(const std::string &s, Args &&... args) {
                lines.emplace_back(indent, fmt::format(s, std::forward<Args>(args)...));
            }
            template <class F>
            void with_block(F &&f) {
                indent++;
                f();
                indent--;
            }
        };
        int temp_counter = 0;
        CodeBlock forward_decls;
        std::vector<CodeBlock> misc_defs;
        std::unordered_map<std::string, type::StructType> structs;
        std::unordered_map<std::string, type::Type> types;
        std::unordered_map<std::string, FunctionRecord> prototypes;
        type::Context type_ctx;
        type::Type process_type(const ast::AST &n);
        type::StructType process_struct_decl(const ast::StructDecl &decl);
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
        void write(std::ostringstream &os, const CodeBlock &block) {
            for (auto &line : block.lines) {
                for (int i = 0; i < line.first + indent; i++)
                    os << "    ";
                os << line.second << "\n";
            }
        }
        [[noreturn]] void error(const SourceLocation &loc, std::string &&msg) {
            throw std::runtime_error(fmt::format("error: {} at {}:{}:{}", msg, loc.filename, loc.line, loc.col));
        }
        void warning(const SourceLocation &loc, std::string &&msg) {
            fmt::print("warning: {} at {}:{}:{}", msg, loc.filename, loc.line, loc.col);
        }
        int eval_const_int(const ast::Expr &e);

      public:
        CodeGenerator();
        std::string generate(const BuildConfig &config_, const Module &module_);
        void add_typedef(const std::string &type, const std::string &def);
        virtual ~CodeGenerator()=default;
    };

    std::unique_ptr<CodeGenerator> cpp_generator();
    std::unique_ptr<CodeGenerator> cuda_generator();
} // namespace akari::asl