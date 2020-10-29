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
namespace akari::render {
    class AKR_EXPORT SceneGraphNode : public sdl::Object {
      public:
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {}
        virtual void commit() {}
        virtual const char *description() { return "unknown"; }
        typedef std::shared_ptr<SceneGraphNode> (*CreateFunc)(void);
    };

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