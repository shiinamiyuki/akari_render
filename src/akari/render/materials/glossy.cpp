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

#include <akari/render/scenegraph.h>
#include <akari/render/texture.h>
#include <akari/render/material.h>
#include <akari/core/color.h>
#include <akari/shaders/common.h>
namespace akari::render {
    using namespace shader;
    class GlossyMaterial : public Material {
      public:
        GlossyMaterial(const Texture *color, const Texture *roughness) : color(color), roughness(roughness) {}
        const Texture *color;
        const Texture *roughness;
        BSDFClosure *evaluate(MaterialEvalContext &ctx) const override {
            auto R = color->evaluate(ctx.sp);
            auto r = roughness->evaluate(ctx.sp)[0];
            return ctx.allocator->new_object<MicrofacetReflection>(R, r);
        }
    };

    class GlossyMaterialNode final : public MaterialNode {
        std::shared_ptr<TextureNode> color;
        std::shared_ptr<TextureNode> roughness;

      public:
        GlossyMaterialNode() { roughness = create_constant_texture_rgb(Vec3(0.001)); }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = resolve_texture(value);
            } else if (field == "roughness") {
                roughness = resolve_texture(value);
            }
        }
        Material *create_material(Allocator<> *allocator) override {
            return allocator->new_object<GlossyMaterial>(color->create_texture(allocator),
                                                         roughness->create_texture(allocator));
        }
    };
    AKR_EXPORT_NODE(GlossyMaterial, GlossyMaterialNode)
} // namespace akari::render