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
#include <akari/core/plugin.h>
#include <akari/core/parser.h>
#include <akari/core/file.h>
namespace akari::render {
    template <typename T>
    struct ObjectCache {
        template <class F>
        T &get_cached_or(F &&f) {
            if (!_storage) {
                _storage = f();
            }
            return _storage.value();
        }
        void invalidate() { _storage.reset(); }

      private:
        std::optional<T> _storage;
    };
    class SceneGraphNode;
    class TraversalCallback {
      public:
        friend class SceneGraphNode;
        virtual void enter(SceneGraphNode *) const {}
        virtual void leave(SceneGraphNode *) const {}
    };
    class NamedNode;
    class AKR_EXPORT SceneGraphNode : public sdl::Object {
      protected:
        virtual void do_traverse(TraversalCallback *cb) {}

      public:
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {}
        virtual bool is_named() const { return false; }
        virtual void on_parameter_change(const std::string &field, const sdl::Value &value) {}
        virtual void commit() {}
        virtual const char *description() { return "unknown"; }
        virtual void traverse(TraversalCallback *cb) {
            cb->enter(this);
            do_traverse(cb);
            cb->leave(this);
        }
        virtual void finalize() {}

        template <typename T>
        inline std::shared_ptr<const T> cast() const;

        template <typename T>
        inline std::shared_ptr<T> cast();

        typedef std::shared_ptr<SceneGraphNode> (*CreateFunc)(void);
    };
    class AKR_EXPORT NamedNode : public SceneGraphNode {
        std::string name_;
        std::shared_ptr<SceneGraphNode> underlying_;

      public:
        NamedNode(std::string name, std::shared_ptr<SceneGraphNode> underlying_)
            : name_(name), underlying_(underlying_) {}
        const std::string &name() { return name_; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            return underlying_->object_field(parser, ctx, field, value);
        }
        void commit() override { underlying_->commit(); }
        const char *description() override { return underlying_->description(); }
        void traverse(TraversalCallback *cb) override { underlying_->traverse(cb); }
        void finalize() override { underlying_->finalize(); }
        bool is_named() const final override { return true; }
        std::shared_ptr<SceneGraphNode> underlying() { return underlying_; }
    };

    template <typename T>
    inline std::shared_ptr<const T> SceneGraphNode::cast() const {
        if (is_named()) {
            return dyn_cast<const NamedNode>(shared_from_this())->underlying();
        } else {
            return dyn_cast<const T>(shared_from_this());
        }
    }

    template <typename T>
    inline std::shared_ptr<T> SceneGraphNode::cast() {
        if (is_named()) {
            return dyn_cast<T>(dyn_cast<NamedNode>(shared_from_this())->underlying());
        } else {
            return dyn_cast<T>(shared_from_this());
        }
    }
    template <typename U, typename T>
    std::shared_ptr<U> sg_dyn_cast(const std::shared_ptr<T> &_p) {
        auto p = dyn_cast<SceneGraphNode>(_p);
        static_assert(std::is_base_of_v<SceneGraphNode, U>);
        return p == nullptr ? nullptr : p->template cast<U>();
    }
    class AKR_EXPORT SceneGraphParser : public sdl::Parser {
      public:
        virtual void register_node(const std::string &name, SceneGraphNode::CreateFunc) = 0;
        static std::shared_ptr<SceneGraphParser> create_parser();
    };
    class CameraNode;
    class MaterialNode;
    class TextureNode;
    class MeshNode;

    template <typename A>
    A load(const sdl::Value &v) {
        using T = typename vec_trait<A>::value_type;
        constexpr int N = vec_trait<A>::size;
        AKR_ASSERT_THROW(v.is_array());
        AKR_ASSERT_THROW(v.size() == size_t(N));
        A a;
        for (int i = 0; i < N; i++) {
            a[i] = v.at(i).get<T>().value();
        }
        return a;
    }
#define AKR_EXPORT_NODE(PLUGIN, CLASS) AKR_EXPORT_PLUGIN(PLUGIN, CLASS, akari::render::SceneGraphNode)
} // namespace akari::render