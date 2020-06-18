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

#include <unordered_map>
#include <akari/compute/transform.hpp>
#include <akari/core/akari.h>

#include "letlist.hpp"

namespace akari::compute::transform {
    using namespace ir;

    class ToAnf : public TransformPass {
        std::unordered_map<std::shared_ptr<ir::Node>, std::shared_ptr<ir::VarNode>> var_map;

      public:
        // call{op, args...} -> let v = call{op', args'...} in ...
        NodePtr transform(const NodePtr &root) override {
            return LetList::with([=](LetList &ll) { return do_transform(root, ll); });
        }

        NodePtr do_transform(const NodePtr &node, LetList &ll) {
            if (node->isa<ConstantNode>() || node->isa<VarNode>() || node->isa<PrimitiveNode>()) {
                return node;
            }
            if (node->isa<LetNode>()) {
                auto let = node->cast<LetNode>();
                ll.push(let->var(), do_transform(let->value(), ll));
                return do_transform(let->body(), ll);
            }
            if (var_map.count(node)) {
                return var_map.at(node);
            }
            VarNodePtr var;
            if (node->isa<CallNode>()) {
                auto cnode = node->cast<CallNode>();
                auto op = do_transform(cnode->op(), ll);
                std::vector<NodePtr> args;
                for (auto &a : cnode->args()) {
                    args.emplace_back(do_transform(a, ll));
                }
                var = ll.push(std::make_shared<CallNode>(op, args));
            } else if (node->isa<FunctionNode>()) {
                auto func = node->cast<FunctionNode>();
                auto params = func->parameters();
                auto body = LetList::with([=](LetList &ll) { return do_transform(func->body(), ll); });
                var = ll.push(std::make_shared<FunctionNode>(params, body));
            } else if (node->isa<IfNode>()) {
                auto if_ = node->cast<IfNode>();
                var = ll.push(std::make_shared<IfNode>(do_transform(if_->cond(), ll), do_transform(if_->then(), ll),
                                                       do_transform(if_->else_(), ll)));
            } else {
                AKR_ASSERT(false);
            }
            var_map.emplace(node, var);
            return var;
        }
    };

    AKR_EXPORT std::shared_ptr<TransformPass> convert_to_anf(){
        return std::make_shared<ToAnf>();
    }
} // namespace akari::compute::transform