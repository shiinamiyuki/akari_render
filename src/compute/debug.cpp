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
#include <akari/core/akari.h>
#include <akari/compute/debug.h>
#include <magic_enum.hpp>
namespace akari::compute::debug {
    using namespace ir;
    AKR_EXPORT std::string to_text(const ir::Node &node) {
        struct DumpVisitor {
            std::string out;

            void emit(int level, const std::string &s) { out.append(std::string(level, ' ')).append(s); }
            void emit(const std::string &s) { out.append(s); }
            void recurse(const Node &node, int level) {
                if (node->isa<Constant>()) {
                    auto val = node->cast<Constant>()->value();
                    std::visit(
                        [=](auto &&arg) {
                            using T = std::decay_t<decltype(arg)>;
                            if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float> ||
                                          std::is_same_v<T, double>) {
                                emit(std::to_string(arg));
                            } else {
                                emit("#constant");
                            }
                        },
                        val);
                } else if (node->isa<Primitive>()) {
                    emit(std::string(magic_enum::enum_name(node->cast<Primitive>()->primitive())).data());
                } else if (node->isa<Var>()) {
                    emit(std::string("%") + node->cast<Var>()->name());
                } else if (node->isa<Function>()) {
                    emit("fn (");
                    auto func = node->cast<Function>();
                    for (auto &i : func->parameters()) {
                        recurse(i, level);
                        emit(" ");
                    }
                    emit("){\n");
                    recurse(func->body(), level + 1);
                    emit("\n");
                    emit(level, "}");
                } else if (node->isa<Call>()) {
                    auto cnode = node->cast<Call>();
                    recurse(cnode->op(), level);
                    emit("(");
                    for (size_t i = 0; i < cnode->args().size(); i++) {
                        recurse(cnode->args()[i], level);
                        if (i + 1 != cnode->args().size()) {
                            emit(", ");
                        }
                    }
                    emit(")");
                } else if (node->isa<If>()) {
                    auto if_ = node->cast<If>();
                    emit(level, "if (");
                    recurse(if_->cond(), level);
                    emit("){\n");
                    recurse(if_->then(), level++);
                    emit(level, "} else {\n");
                    recurse(if_->else_(), level++);
                    emit(level, "}\n");
                } else if (node->isa<Let>()) {
                    auto let = node->cast<Let>();
                    emit(level, "let ");
                    recurse(let->var(), level);
                    emit(level, " = ");
                    recurse(let->value(), level);
                    emit(";\n");
                    if (!let->body()->isa<Let>()) {
                        emit(level, "");
                    }
                    recurse(let->body(), level);
                } else {
                    AKR_ASSERT(false);
                }
            }
        };
        DumpVisitor vis;
        vis.recurse(node, 0);
        return vis.out;
    }
} // namespace akari::compute::debug