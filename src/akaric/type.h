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
namespace akari::asl::type {
    enum Qualifier {
        none = 0,
        in = 1,
        out = 1 << 1,
        inout = in | out,
        const_ = 1 << 2,
        read = 1 << 3,
        write = 1 << 4,
        uniform = 1 << 5,
    };
    class TypeNode : public Base {
      public:
        Qualifier qualifier = Qualifier::none;
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
    class VoidTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(VoidTypeNode)
    };
    using VoidType = std::shared_ptr<VoidTypeNode>;
    class PrimitiveTypeNode : public TypeNode {
      public:
        virtual size_t width() const { return 0; }
        AKR_DECL_TYPENODE(PrimitiveTypeNode)
    };
    using PrimitiveType = std::shared_ptr<PrimitiveTypeNode>;
    class PrimitiveTypeFloat64 : public PrimitiveTypeNode {
      public:
        AKR_DECL_TYPENODE(PrimitiveTypeFloat64)
        bool is_float() const override { return true; }
        bool is_int() const override { return false; }
        bool is_signed_int() const override { return false; }
        bool is_aggregate() const override { return false; }
        size_t width() const override { return 64; }
    };
    class PrimitiveTypeFloat32 : public PrimitiveTypeFloat64 {
      public:
        AKR_DECL_TYPENODE(PrimitiveTypeFloat32)
        bool is_float() const override { return true; }
        bool is_int() const override { return false; }
        bool is_signed_int() const override { return false; }
        bool is_aggregate() const override { return false; }
        size_t width() const override { return 32; }
    };

    class PrimitiveTypeInt32 : public PrimitiveTypeNode {
      public:
        AKR_DECL_TYPENODE(PrimitiveTypeInt32)
        bool is_float() const override { return false; }
        bool is_int() const override { return true; }
        bool is_signed_int() const override { return true; }
        bool is_aggregate() const override { return false; }
        size_t width() const override { return 32; }
    };
    class PrimitiveTypeUint32 : public PrimitiveTypeNode {
      public:
        AKR_DECL_TYPENODE(PrimitiveTypeUint32)
        bool is_float() const override { return false; }
        bool is_int() const override { return true; }
        bool is_signed_int() const override { return false; }
        bool is_aggregate() const override { return false; }
        size_t width() const override { return 32; }
    };
    class PrimitiveTypeBoolean : public PrimitiveTypeUint32 {
      public:
        AKR_DECL_TYPENODE(PrimitiveTypeBoolean)
        bool is_float() const override { return false; }
        bool is_int() const override { return true; }
        bool is_signed_int() const override { return false; }
        bool is_aggregate() const override { return false; }
        size_t width() const override { return 8; }
    };
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
    class VectorTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(VectorTypeNode)
        Type element_type;
        int count;
        bool is_float() const override { return element_type->is_float(); }
        bool is_int() const override { return element_type->is_int(); }
        bool is_signed_int() const override { return element_type->is_signed_int(); }
        bool is_aggregate() const override { return false; }
    };
    using VectorType = std::shared_ptr<VectorTypeNode>;
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
    class TupleTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(TupleTypeNode)
        TupleTypeNode(std::vector<Type> element_types) : element_types(element_types) {}
        std::vector<Type> element_types;
    };
    using TupleType = std::shared_ptr<TupleTypeNode>;

    class OptionalTypeNode : public TypeNode {
      public:
        AKR_DECL_TYPENODE(OptionalTypeNode)
        OptionalTypeNode(Type element_types) : element_type(element_types) {}
        Type element_type;
    };
    using OptionalType = std::shared_ptr<OptionalTypeNode>;
    struct AnnotatedType {
        Type type;
        Qualifier qualifier = Qualifier::none;
        AnnotatedType(Type type) : type(type) {}
        AnnotatedType(Type type, Qualifier qualifier) : type(type), qualifier(qualifier) {}
    };
    template <typename T1, typename T2>
    struct PairHash {
        size_t operator()(const std::pair<T1, T2> &pair) const{
            return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
        }
    };
    class Context {
        struct TupleHash {
            size_t operator()(const std::vector<type::Type> &types) const {
                size_t s = 0;
                for (auto &t : types) {
                    s ^= std::hash<type::Type>()(t);
                }
                return s;
            }
        };
        struct TupleEqual {
            bool operator()(const std::vector<type::Type> &a, const std::vector<type::Type> &b) const {
                if (a.size() != b.size())
                    return false;
                for (size_t i = 0; i < a.size(); i++) {
                    if (a[i] != b[i])
                        return false;
                }
                return true;
            }
        };
        std::unordered_map<std::pair<type::Type, int>, type::ArrayType, PairHash<type::Type, int>> array_cache;
        std::unordered_map<std::vector<type::Type>, type::TupleType, TupleHash, TupleEqual> tuple_cache;

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
        type::TupleType make_tuple(const std::vector<type::Type> &types) {
            auto it = tuple_cache.find(types);
            if (tuple_cache.end() == it) {
                tuple_cache[types] = std::make_shared<TupleTypeNode>(types);
                return tuple_cache[types];
            }
            return it->second;
        }
    };
} // namespace akari::asl::type