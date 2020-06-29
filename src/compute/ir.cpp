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

#include <akari/compute/ir.hpp>

namespace akari::compute::ir {
    size_t generate_id() {
        static size_t i = 0;
        return i++;
    }
    static PrimitiveType boolean = std::make_shared<PrimitiveTypeNode>(PrimitiveTy::boolean);
    static PrimitiveType int32 = std::make_shared<PrimitiveTypeNode>(PrimitiveTy::int32);
    static PrimitiveType float32 = std::make_shared<PrimitiveTypeNode>(PrimitiveTy::float32);
    static PrimitiveType float64 = std::make_shared<PrimitiveTypeNode>(PrimitiveTy::float64);
    PrimitiveType get_primitive_type(PrimitiveTy ty) {
        switch (ty) {
        case PrimitiveTy::boolean:
            return boolean;
        case PrimitiveTy::int32:
            return int32;
        case PrimitiveTy::float32:
            return float32;
        case PrimitiveTy::float64:
            return float64;
        default:
            return nullptr;
        }
    }
} // namespace akari::compute::ir