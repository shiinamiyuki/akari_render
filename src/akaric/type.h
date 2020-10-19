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

#include <akaric/base.h>
#include <tuple>
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
    std::string type_name() const { return #Type; }                                                                    \
    bool is_parent_of(const std::shared_ptr<Base> &ptr) const { return ptr->isa<std::shared_ptr<Type>>(); }

    using Type = std::shared_ptr<TypeNode>;
    class OpaqueTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(OpaqueTypeNode)
        OpaqueTypeNode(const std::string &name) : name(name) {}
        std::string name;
    };
    using OpaqueType = std::shared_ptr<OpaqueTypeNode>;
    class PrimitiveTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(PrimitiveTypeNode)
    };
    using PrimitiveType = std::shared_ptr<PrimitiveTypeNode>;
    class VoidTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(VoidTypeNode)
    };
    using VoidType = std::shared_ptr<VoidTypeNode>;
    extern PrimitiveType boolean;
    extern PrimitiveType int32;
    extern PrimitiveType uint32;
    extern PrimitiveType float32;
    extern PrimitiveType float64;
    extern VoidType void_;

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
    class FunctionTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(FunctionTypeNode)
        Type ret;
        std::vector<Type> args;
    };
    using FunctionType = std::shared_ptr<FunctionTypeNode>;
    enum class Qualifier { none = 0, in = 1, out = 2, inout = 3, constant = 4 };
    class QualifiedTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(QualifiedTypeNode)
        QualifiedTypeNode(Type e, Qualifier qualifier = Qualifier::none) : element_type(e), qualifier(qualifier) {}
        Type element_type;
        Qualifier qualifier;
    };
    using QualifiedType = std::shared_ptr<QualifiedTypeNode>;
    class VectorTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(VectorTypeNode)
        VectorTypeNode(Type e, int count) : element_type(e), count(count) {}
        Type element_type;
        int count;
        bool is_float() const override { return element_type->is_float(); }
        bool is_int() const override { return element_type->is_int(); }
        bool is_signed_int() const override { return element_type->is_signed_int(); }
        bool is_aggregate() const override { return false; }
    };
    using VectorType = std::shared_ptr<VectorTypeNode>;
    class MatrixTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(MatrixTypeNode)
        MatrixTypeNode(Type e, int cols, int rows) : element_type(e), rows(rows), cols(cols) {}
        Type element_type;
        int rows, cols;
        bool is_float() const override { return element_type->is_float(); }
        bool is_int() const override { return element_type->is_int(); }
        bool is_signed_int() const override { return element_type->is_signed_int(); }
        bool is_aggregate() const override { return false; }
    };
    using MatrixType = std::shared_ptr<MatrixTypeNode>;
    class ArrayTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(ArrayTypeNode)
        ArrayTypeNode(Type e, int l) : element_type(e), length(l) {}
        Type element_type;
        int length = -1;
        bool is_float() const override { return element_type->is_float(); }
        bool is_int() const override { return element_type->is_int(); }
        bool is_signed_int() const override { return element_type->is_signed_int(); }
        bool is_aggregate() const override { return false; }
    };
    using ArrayType = std::shared_ptr<ArrayTypeNode>;

    class UniformTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(UniformTypeNode)
        UniformTypeNode(Type element_types) : element_type(element_types) {}
        Type element_type;
    };
    using UniformType = std::shared_ptr<UniformTypeNode>;
    template <typename T1, typename T2>
    struct PairHash {
        size_t operator()(const std::pair<T1, T2> &pair) const {
            return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
        }
    };
    template <typename T1, typename T2, typename T3>
    struct TupleHash {
        size_t operator()(const std::tuple<T1, T2, T3> &t) const {
            return std::hash<T1>()(std::get<0>(t)) ^ std::hash<T2>()(std::get<1>(t)) ^ std::hash<T3>()(std::get<2>(t));
        }
    };
    template <typename T1, typename T2, typename T3>
    struct TupleEq {
        bool operator()(const std::tuple<T1, T2, T3> &t1, const std::tuple<T1, T2, T3> &t2) const {
            const auto &[x1, y1, z1] = t1;
            const auto &[x2, y2, z2] = t2;
            return x1 == x2 && y1 == y2 && z1 == z2;
        }
    };
    class Context {

        std::unordered_map<std::pair<type::Type, int>, type::VectorType, PairHash<type::Type, int>> vector_cache;
        std::unordered_map<std::tuple<type::Type, int, int>, type::MatrixType, TupleHash<type::Type, int, int>,
                           TupleEq<type::Type, int, int>>
            mat_cache;
        std::unordered_map<type::Type, type::QualifiedType> q_cache;
        std::unordered_map<std::pair<type::Type, int>, type::ArrayType, PairHash<type::Type, int>> array_cache;

      public:
        type::ArrayType make_array(type::Type t, int length) {
            auto pair = std::make_pair(t, length);
            auto it = array_cache.find(pair);
            if (array_cache.end() == it) {
                array_cache[pair] = std::make_shared<ArrayTypeNode>(t, length);
                return array_cache[pair];
            }
            return it->second;
        }
        type::VectorType make_vector(type::Type t, int length) {
            auto pair = std::make_pair(t, length);
            auto it = vector_cache.find(pair);
            if (vector_cache.end() == it) {
                vector_cache[pair] = std::make_shared<VectorTypeNode>(t, length);
                return vector_cache[pair];
            }
            return it->second;
        }
        type::MatrixType make_mat(type::Type t, int n, int m) {
            auto tuple = std::make_tuple(t, n, m);
            auto it = mat_cache.find(tuple);
            if (mat_cache.end() == it) {
                mat_cache[tuple] = std::make_shared<MatrixTypeNode>(t, n, m);
                return mat_cache[tuple];
            }
            return it->second;
        }
        type::QualifiedType make_qualified(type::Type t, Qualifier qualifer) {
            auto it = q_cache.find(t);
            if (q_cache.end() == it) {
                q_cache[t] = std::make_shared<QualifiedTypeNode>(t, qualifer);
                return q_cache[t];
            }
            return it->second;
        }
    };
    inline type::Type remove_qualifier(type::Type ty) {
        if (auto m = ty->cast<type::QualifiedType>()) {
            return m->element_type;
        }
        return ty;
    }

    inline type::Type value_type(type::Type ty) {
        if (auto v = ty->cast<type::VectorType>()) {
            return v->element_type;
        }
        if (auto a = ty->cast<type::ArrayType>()) {
            return a->element_type;
        }
        if (auto a = ty->cast<type::MatrixType>()) {
            return a->element_type;
        }
        return ty;
    }
} // namespace akari::asl::type