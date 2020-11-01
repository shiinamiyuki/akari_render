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
#include <akari/render/common.h>
namespace akari::render {
    class SpecularMaterial : public Material {
      public:
        enum Mode { Reflect, Refract, Blend };
        SpecularMaterial(std::shared_ptr<const Texture> color, Float eta, Mode mode)
            : color(color), eta(eta), mode(mode) {}
        std::shared_ptr<const Texture> color;
        Float eta;
        Mode mode;
        BSDFClosure *evaluate(MaterialEvalContext &ctx) const override {
            auto R = color->evaluate(ctx.sp);
            if (mode == Mode::Reflect) {
                return ctx.allocator.new_object<SpecularReflection>(R);
            } else if (mode == Mode::Refract) {
                return ctx.allocator.new_object<SpecularTransmission>(R, eta);
            } else {
                return ctx.allocator.new_object<FresnelSpecular>(R, R, 1.0, eta);
            }
        }
        Spectrum albedo(const ShadingPoint &sp) const override {
            auto R = color->evaluate(sp);
            return R;
        }
        Float tr(const ShadingPoint &sp) const override { return color->tr(sp); }
    };

    class SpecularMaterialNode final : public MaterialNode {
        std::shared_ptr<TextureNode> color;
        Float eta = 1.33;
        using Mode = SpecularMaterial::Mode;
        Mode mode = Mode::Reflect;

      public:
        SpecularMaterialNode() {}
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = resolve_texture(value);
            } else if (field == "eta") {
                eta = value.get<float>().value();
            } else if (field == "mode") {
                auto s = value.get<std::string>().value();
                if (s == "reflect") {
                    mode = Mode::Reflect;
                } else if (s == "refract") {
                    mode = Mode::Refract;
                } else if (s == "blend") {
                    mode = Mode::Blend;
                } else {
                    throw std::runtime_error(fmt::format("unknown mode {}", s));
                }
            }
        }
        void finalize() override { color->finalize(); }
        std::shared_ptr<const Material> create_material(Allocator<> allocator) override {
            return make_pmr_shared<SpecularMaterial>(allocator, color->create_texture(allocator), eta, mode);
        }
    };
    AKR_EXPORT_NODE(SpecularMaterial, SpecularMaterialNode)
} // namespace akari::render