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
namespace akari::compute::debug{
    using namespace ir;
    AKR_EXPORT std::string to_text(const ir::NodePtr & node){
        struct DumpVisitor {
            std::string out;

            void emit(int level, const std::string & s){
                out.append(std::string(level, ' ')).append(s);
            }
            void emit(const std::string & s){
                out.append(s);
            }
            void recurse(const NodePtr & node, int level){
                if(node->isa<ConstantNode>()){
                    auto val = node->cast<ConstantNode>()->value();
                    std::visit([=](auto && arg){
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, int>
                            || std::is_same_v<T, float>
                            || std::is_same_v<T, double>){
                            emit(std::to_string(arg));
                        }else{
                            emit("#constant");
                        }
                    }, val);
                }else if(node->isa<PrimitiveNode>()){
                    emit(std::string(magic_enum::enum_name(node->cast<PrimitiveNode>()->primitive())).data());
                }else if(node->isa<VarNode>()){
                    emit(std::string("%") + node->cast<VarNode>()->name());
                }else if(node->isa<FunctionNode>()){
                    emit("fn (");
                    auto func = node->cast<FunctionNode>();
                    for(auto & i: func->parameters()){
                        recurse(i, level);
                        emit(" ");
                    }
                    emit("){\n");
                    recurse(func->body(),level+1);
                    emit("\n");
                    emit(level, "}");
                }else if(node->isa<CallNode>()){
                    auto cnode = node->cast<CallNode>();
                    recurse(cnode->op(), level);
                    emit("(");
                    for(size_t i = 0; i< cnode->args().size();i++){
                        recurse(cnode->args()[i], level);
                        if(i + 1!= cnode->args().size()){
                            emit(", ");
                        }
                    }
                    emit(")");
                }else if(node->isa<IfNode>()){
                    auto if_  = node->cast<IfNode>();
                    emit(level, "if (");
                    recurse(if_->cond(), level);
                    emit("){\n");
                    recurse(if_->then(), level++);
                    emit(level, "} else {\n");
                    recurse(if_->else_(), level++);
                    emit(level, "}\n");
                }else if(node->isa<LetNode>()){
                    auto let = node->cast<LetNode>();
                    emit(level, "let ");
                    recurse(let->var(), level);
                    emit(level, " = ");
                    recurse(let->value(), level);
                    emit(";\n");
                    if(!let->body()->isa<LetNode>()){
                        emit(level, "");
                    }
                    recurse(let->body(), level);
                }else{
                    AKR_ASSERT(false);
                }
            }
        };
        DumpVisitor vis;
        vis.recurse(node, 0);
        return vis.out;
    }
}