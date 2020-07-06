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

#include <akari/asl/type.h>

namespace akari::asl {
    namespace type {
        class PrimitiveTypeBoolean : public PrimitiveTypeNode {
          public:
            AKR_DECL_TYPENODE(PrimitiveTypeBoolean)
            bool is_float() const { return false; }
            bool is_int() const { return true; }
            bool is_signed_int() const { return false; }
            bool is_aggregate() const { return false; }
        };
        class PrimitiveTypeInt32 : public PrimitiveTypeNode {
          public:
            AKR_DECL_TYPENODE(PrimitiveTypeInt32)
            bool is_float() const { return false; }
            bool is_int() const { return true; }
            bool is_signed_int() const { return true; }
            bool is_aggregate() const { return false; }
        };
        class PrimitiveTypeUint32 : public PrimitiveTypeNode {
          public:
            AKR_DECL_TYPENODE(PrimitiveTypeUint32)
            bool is_float() const { return false; }
            bool is_int() const { return true; }
            bool is_signed_int() const { return false; }
            bool is_aggregate() const { return false; }
        };
        class PrimitiveTypeFloat32 : public PrimitiveTypeNode {
          public:
            AKR_DECL_TYPENODE(PrimitiveTypeUint32)
            bool is_float() const { return true; }
            bool is_int() const { return false; }
            bool is_signed_int() const { return false; }
            bool is_aggregate() const { return false; }
        };
        class PrimitiveTypeFloat64 : public PrimitiveTypeNode {
          public:
            AKR_DECL_TYPENODE(PrimitiveTypeFloat64)
            bool is_float() const { return true; }
            bool is_int() const { return false; }
            bool is_signed_int() const { return false; }
            bool is_aggregate() const { return false; }
        };
        AKR_EXPORT PrimitiveType boolean = std::make_shared<PrimitiveTypeBoolean>();
        AKR_EXPORT PrimitiveType int32 = std::make_shared<PrimitiveTypeInt32>();
        AKR_EXPORT PrimitiveType uint32 = std::make_shared<PrimitiveTypeUint32>();
        AKR_EXPORT PrimitiveType float32 = std::make_shared<PrimitiveTypeFloat32>();
        AKR_EXPORT PrimitiveType float64 = std::make_shared<PrimitiveTypeBoolean>();
    } // namespace type
} // namespace akari::asl