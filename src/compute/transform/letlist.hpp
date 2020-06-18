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
#include <unordered_map>
#include <akari/compute/transform.hpp>
#include <akari/core/akari.h>

namespace akari::compute::transform {
    struct LetList {
        std::list<std::pair<ir::VarNodePtr, ir::NodePtr>> list;
        ir::NodePtr construct(const ir::NodePtr &body) {
            ir::NodePtr expr = body;
            for (auto it = list.rbegin(); it != list.rend(); it++) {
                expr = std::make_shared<ir::LetNode>(it->first, it->second, expr);
            }
            return expr;
        }
        void push(ir::VarNodePtr var, ir::NodePtr val) { list.emplace_back(var, val); }
        ir::VarNodePtr push(ir::NodePtr val) {
            auto var = std::make_shared<ir::VarNode>(ir::generate_id());
            push(var, val);
            return var;
        }
        template <typename F> static ir::NodePtr with(F &&f) {
            LetList ll;
            return ll.construct(f(ll));
        }
    };
}