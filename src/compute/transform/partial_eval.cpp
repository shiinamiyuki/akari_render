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
    struct PStatic;
    struct Empty{};
    using SFunction = std::function<PStatic(const std::vector<PStatic>&)>;
    struct SValue : std::variant<ConstantNodePtr,SFunction, Empty>{
        using std::variant<ConstantNodePtr,SFunction, Empty>::variant;

    };
    struct PStatic {

    };
    class PartialEvaluate : public TransformPass {
    public:
        NodePtr transform(const NodePtr &root) override {
            return nullptr;
        }
    };

    AKR_EXPORT std::shared_ptr<TransformPass> partial_eval(){
        return std::make_shared<PartialEvaluate>();
    }
}