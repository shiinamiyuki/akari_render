

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
#include <akari/core/math.h>
#include <akari/core/variant.h>
#include <akari/render/scenegraph.h>
#include <akari/render/common.h>
#include <akari/core/memory.h>
#include <akari/render/sampler.h>
#include <akari/render/texture.h>
#include <akari/render/closure.h>
#include <optional>

namespace akari::render {
    class Material;
    class DiffuseMaterial;
    class GlossyMaterial;
    class EmissiveMaterial;
    class MixMaterial;
    struct SurfaceInteraction;

    struct BSDFSampleContext {
        const vec2 u1;
        const Vec3 wo;

        BSDFSampleContext(const vec2 &u1, const Vec3 &wo) : u1(u1), wo(wo) {}
    };

    class BSDF {
        const BSDFClosure *closure_ = nullptr;
        Vec3 ng, ns;
        Frame frame;
        Float choice_pdf = 1.0f;

      public:
        BSDF() = default;
        explicit BSDF(const Vec3 &ng, const Vec3 &ns) : ng(ng), ns(ns) {
            frame = Frame(ns);
            (void)this->ng;
            (void)this->ns;
        }
        bool null() const { return closure_ == nullptr; }
        void set_closure(const BSDFClosure *closure) { closure_ = closure; }
        void set_choice_pdf(Float pdf) { choice_pdf = pdf; }
        const BSDFClosure &closure() const { return *closure_; }
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            auto pdf = closure().evaluate_pdf(frame.world_to_local(wo), frame.world_to_local(wi));
            return pdf * choice_pdf;
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const {
            auto f = closure().evaluate(frame.world_to_local(wo), frame.world_to_local(wi));
            return f;
        }

        [[nodiscard]] BSDFType type() const { return closure().type(); }
        [[nodiscard]] bool match_flags(BSDFType flag) const { return closure().match_flags(flag); }
        std::optional<BSDFSample> sample(const BSDFSampleContext &ctx) const {
            auto wo = frame.world_to_local(ctx.wo);
            if (auto sample = closure().sample(ctx.u1, wo)) {
                sample->wi = frame.local_to_world(sample->wi);
                sample->pdf *= choice_pdf;
                return sample;
            }
            return std::nullopt;
        }
    };
    struct MaterialEvalContext {
        Allocator<> allocator;
        vec2 u1;
        ShadingPoint sp;
        Vec3 ng, ns;
        MaterialEvalContext(Allocator<> allocator, Sampler *sampler, const vec2 &texcoords, const Vec3 &ng,
                            const Vec3 &ns)
            : allocator(allocator), u1(sampler->next2d()), sp(texcoords), ng(ng), ns(ns) {}
    };
    class EmissiveMaterial;
    class AKR_EXPORT Material {
      public:
        virtual inline const BSDFClosure *evaluate(MaterialEvalContext &ctx) const = 0;
        virtual inline Spectrum albedo(const ShadingPoint &sp) const = 0;

        virtual Float tr(const ShadingPoint &sp) const = 0;
        virtual const EmissiveMaterial *as_emissive() const { return nullptr; }
        BSDF get_bsdf(MaterialEvalContext &ctx) const;
    };
    class LightNode;
    class AKR_EXPORT EmissiveMaterial : public Material {
      public:
        std::shared_ptr<LightNode> light;
        EmissiveMaterial(std::shared_ptr<LightNode> light) : light(light) {}
        const EmissiveMaterial *as_emissive() const override { return this; }
        virtual inline const BSDFClosure *evaluate(MaterialEvalContext &ctx) const { return nullptr; }
        virtual inline Spectrum albedo(const ShadingPoint &sp) const { return Spectrum(0.0); }

        virtual Float tr(const ShadingPoint &sp) const { return 0.0; }
    };
    class EmissiveMaterialNode;
    class AKR_EXPORT MaterialNode : public SceneGraphNode {
      public:
        virtual [[nodiscard]] std::shared_ptr<const Material> create_material(Allocator<>) = 0;
        virtual [[nodiscard]] std::shared_ptr<EmissiveMaterialNode> as_emissive() { return nullptr; }
    };

    class AKR_EXPORT EmissiveMaterialNode : public MaterialNode {
      public:
        [[nodiscard]] std::shared_ptr<EmissiveMaterialNode> as_emissive() override {
            return dyn_cast<EmissiveMaterialNode>(shared_from_this());
        }
        [[nodiscard]] virtual std::shared_ptr<LightNode> light() = 0;
    };
    AKR_EXPORT std::shared_ptr<EmissiveMaterialNode> create_emissive_material();
    inline std::shared_ptr<TextureNode> resolve_texture(const sdl::Value &value) {
        if (value.is_array()) {
            if (value.size() == 3)
                return create_constant_texture_rgb(load<Color3f>(value));
            else {
                vec4 rgba = load<vec4>(value);
                return create_constant_texture_rgba(RGBA(Color3f(rgba.x, rgba.y, rgba.z), rgba.w));
            }
        } else if (value.is_number()) {
            return create_constant_texture_rgb(Color3f(value.get<float>().value()));
        } else if (value.is_string()) {
            auto path = value.get<std::string>().value();
            return create_image_texture(path);
        } else {
            AKR_ASSERT_THROW(value.is_object());
            auto tex = dyn_cast<TextureNode>(value.object());
            AKR_ASSERT_THROW(tex);
            return tex;
        }
    }

    class MixMaterial : public Material {
      public:
        MixMaterial(std::shared_ptr<const Texture> fraction, std::shared_ptr<const Material> mat_A,
                    std::shared_ptr<const Material> mat_B)
            : fraction(fraction), mat_A(mat_A), mat_B(mat_B) {}
        std::shared_ptr<const Texture> fraction = nullptr;
        std::shared_ptr<const Material> mat_A = nullptr;
        std::shared_ptr<const Material> mat_B = nullptr;
        const BSDFClosure *evaluate(MaterialEvalContext &ctx) const override {
            auto closure = ctx.allocator.new_object<MixBSDF>(fraction->evaluate(ctx.sp)[0], mat_A->evaluate(ctx),
                                                             mat_B->evaluate(ctx));
            return closure;
        }
        Spectrum albedo(const ShadingPoint &sp) const override {
            return lerp(mat_A->albedo(sp), mat_B->albedo(sp), fraction->evaluate(sp)[0]);
        }
        Float tr(const ShadingPoint &sp) const override {
            return lerp(mat_A->tr(sp), mat_B->tr(sp), fraction->evaluate(sp)[0]);
        }
    };
    AKR_EXPORT std::shared_ptr<MaterialNode> create_mix_material();
} // namespace akari::render