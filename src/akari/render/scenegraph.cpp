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

#include <mutex>
#include <akari/render/scenegraph.h>
#include <akari/render/scene.h>
#include <akari/render/camera.h>
#include <akari/render/integrator.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/render/light.h>

namespace akari::render {
    class SceneGraphParserImpl : public SceneGraphParser {
        using NodePlugin = PluginInterface<SceneGraphNode>;
        using NodeManager = PluginManager<SceneGraphNode>;
        NodeManager mgr;
        std::unordered_map<std::string, SceneGraphNode::CreateFunc> registerd_nodes;

      public:
        void register_node(const std::string &name, SceneGraphNode::CreateFunc func) override {
            registerd_nodes.emplace(name, func);
        }
        SceneGraphParserImpl() {
            register_node("ConstantTexture", [] { return dyn_cast<SceneGraphNode>(create_constant_texture()); });
            register_node("ImageTexture", [] { return dyn_cast<SceneGraphNode>(create_image_texture()); });
            register_node("RandomSampler",
                          [] { return dyn_cast<SceneGraphNode>(std::make_shared<RandomSamplerNode>()); });
#define REG_NODE(name, func) register_node(name, [] { return dyn_cast<SceneGraphNode>(func()); })
            REG_NODE("MixMaterial", create_mix_material);
            REG_NODE("EmissiveMaterial", create_emissive_material);
            register_node("Scene", [] { return dyn_cast<SceneGraphNode>(std::make_shared<SceneNode>()); });
            register_node("AOV", [] { return dyn_cast<SceneGraphNode>(make_aov_integrator()); });
        }
        sdl::P<sdl::Object> do_parse_object_creation(sdl::ParserContext &ctx, const std::string &type) override final {
            if (registerd_nodes.find(type) != registerd_nodes.end()) {
                return registerd_nodes.at(type)();
            }
            auto pi = mgr.load_plugin(type);
            if (!pi) {
                throw std::runtime_error("failed to load plugin");
            }
            auto object = pi->make_shared();
            if(!ctx.cur_var.empty()){
                return std::make_shared<NamedNode>(ctx.cur_var, object);
            }
            return object;
        }
    };
    std::shared_ptr<SceneGraphParser> SceneGraphParser::create_parser() {
        static std::shared_ptr<SceneGraphParserImpl> impl;
        static std::once_flag flag;
        std::call_once(flag, [&] { impl = std::make_shared<SceneGraphParserImpl>(); });
        return impl;
    }

} // namespace akari::render