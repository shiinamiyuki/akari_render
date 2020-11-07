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
#include <akari/render/material.h>
#include <akari/core/memory.h>
namespace akari::render {
    BSDF Material::get_bsdf(MaterialEvalContext &ctx) const {
        auto tr_ = tr(ctx.sp);
        bool pass_through = false;
        Float pass_through_pdf = 0.0;
        if (tr_ > 0.0) {
            pass_through = ctx.u1[1] < tr_;
            pass_through_pdf = tr_;
        }
        if (!pass_through) {
            auto closure = evaluate(ctx);
            BSDF bsdf(ctx.ng, ctx.ns);
            bsdf.set_closure(closure);
            if (pass_through_pdf != 0.0) {
                bsdf.set_choice_pdf(1.0f - pass_through_pdf);
            }
            return bsdf;
        } else {
            auto closure = ctx.allocator.new_object<PassThrough>(Spectrum(1));
            BSDF bsdf(ctx.ng, ctx.ns);
            bsdf.set_closure(closure);
            bsdf.set_choice_pdf(pass_through_pdf);
            return bsdf;
        }
    }
    class MixMaterialNode final : public MaterialNode {
        std::shared_ptr<TextureNode> fraction;
        std::shared_ptr<MaterialNode> mat_A;
        std::shared_ptr<MaterialNode> mat_B;

      public:
        AKR_SER_CLASS("MixMaterial")
        AKR_SER(fraction, mat_A, mat_B)
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "fraction" || field == "frac") {
                fraction = resolve_texture(value);
            } else if (field == "first") {
                mat_A = sg_dyn_cast<MaterialNode>(value.object());
                AKR_ASSERT_THROW(mat_A);
            } else if (field == "second") {
                mat_B = sg_dyn_cast<MaterialNode>(value.object());
                AKR_ASSERT_THROW(mat_B);
            }
        }
        void finalize() override {
            fraction->finalize();
            mat_A->finalize();
            mat_B->finalize();
        }
        std::shared_ptr<const Material> create_material(Allocator<> allocator) override {
            return make_pmr_shared<const MixMaterial>(allocator, fraction->create_texture(allocator),
                                                      mat_A->create_material(allocator),
                                                      mat_B->create_material(allocator));
        }
    };

    AKR_EXPORT std::shared_ptr<MaterialNode> create_mix_material() { return std::make_shared<MixMaterialNode>(); }

    class AKR_EXPORT EmissiveMaterialNodeImpl final : public EmissiveMaterialNode {
        std::shared_ptr<LightNode> light_;

      public:
        AKR_SER_CLASS("EmissiveMaterial") [[nodiscard]] std::shared_ptr<LightNode> light() override { return light_; }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "light") {
                light_ = sg_dyn_cast<LightNode>(value.object());
                AKR_ASSERT_THROW(light_);
            }
        }
        std::shared_ptr<const Material> create_material(Allocator<> allocator) override {
            return make_pmr_shared<EmissiveMaterial>(allocator, light_);
        }
        void finalize() override { light_->finalize(); }
    };
    AKR_EXPORT std::shared_ptr<EmissiveMaterialNode> create_emissive_material() {
        return std::make_shared<EmissiveMaterialNodeImpl>();
    }
} // namespace akari::render