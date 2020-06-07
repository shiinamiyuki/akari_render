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
#include <string>
#include <akari/compute/base.hpp>

namespace akari::compute::type {
    class Type : public Base {};
    class ScalarType;
    class TensorType;
    class VectorType;
    class MatrixType;
    class FunctionType;
    class StructType;
    class EnumType;
    class GenericType;
    class TupleType;
    class TypeVisitor {
      public:
        virtual void visit(ScalarType &) {}
        virtual void visit(TensorType &) {}
        virtual void visit(MatrixType &) {}
        virtual void visit(VectorType &) {}
        virtual void visit(FunctionType &) {}
        virtual void visit(StructType &) {}
        virtual void visit(EnumType &) {}
        virtual void visit(GenericType &) {}
    };
#define AKR_DECL_TYPENODE(Type)                                                                                        \
    std::string type_name() const { return #Type; }                                                                    \
    void accept(TypeVisitor &vis) { vis.visit(*this); }

    class GenericType : public Type {
      public:
        AKR_DECL_TYPENODE(GenericType)
    };
     class TensorType : public Type {
      public:
        AKR_DECL_TYPENODE(TensorType)
    };
    class ScalarType : public TensorType {
      public:
        AKR_DECL_TYPENODE(ScalarType)
    };
    class MatrixType : public TensorType {
      public:
        AKR_DECL_TYPENODE(MatrixType)
    };   
    class VectorType : public TensorType {
      public:
        AKR_DECL_TYPENODE(VectorType)
    };
    class FunctionType : public Type {
      public:
        AKR_DECL_TYPENODE(FunctionType)
    };
    class StructType : public Type {
      public:
        AKR_DECL_TYPENODE(StructType)
    };
    class EnumType : public Type {
      public:
        AKR_DECL_TYPENODE(EnumType)
    };
    class TupleType : public Type {
      public:
        AKR_DECL_TYPENODE(TupleType)
    };
    using TypePtr = std::shared_ptr<Type>;
} // namespace akari::compute::type