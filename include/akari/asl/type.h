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
#include <akari/core/akari.h>
#include <akari/core/platform.h>
#include <akari/asl/base.h>
namespace akari::asl::type {
    class TypeNode : public Base {
      public:
        // scalar, vector, ...
        virtual bool is_float() const { return false; }
        virtual bool is_int() const { return false; }
        virtual bool is_signed_int() const { return false; }
        virtual bool is_aggregate() const { return false; }
    };
#define AKR_DECL_TYPENODE(Type)                                                                                        \
    std::string type_name() const { return #Type; }

    using Type = std::shared_ptr<TypeNode>;
    class PrimitiveTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(PrimitiveTypeNode)
    };
    using PrimitiveType = std::shared_ptr<PrimitiveTypeNode>;

    AKR_EXPORT extern PrimitiveType boolean;
    AKR_EXPORT extern PrimitiveType int32;
    AKR_EXPORT extern PrimitiveType uint32;
    AKR_EXPORT extern PrimitiveType float32;
    AKR_EXPORT extern PrimitiveType float64;

    struct StructField {
        std::string name;
        Type type;
        int index;
    };
    class StructTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(StructTypeNode)
        std::string name;
        std::vector<StructField> fields;
        void append(Type ty) { fields.emplace_back(StructField{"", ty, (int)fields.size()}); }
        void append(std::string _name, Type ty) { fields.emplace_back(StructField{_name, ty, (int)fields.size()}); }
    };
    using StructType = std::shared_ptr<StructTypeNode>;
    AKR_EXPORT StructType get_struct_type(const char *);
    AKR_EXPORT void register_struct_type(const char *, StructType);

} // namespace akari::asl::type