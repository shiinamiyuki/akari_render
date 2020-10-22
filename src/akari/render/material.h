

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
#include <akari/render/scenegraph.h>
#include <akari/shaders/common.h>
#include <akari/core/memory.h>
#include <akari/render/sampler.h>
#include <akari/render/interaction.h>
#include <akari/render/texture.h>
#include <optional>

namespace akari::render {
    enum BSDFType : int {
        BSDF_NONE = 0u,
        BSDF_REFLECTION = 1u << 0u,
        BSDF_TRANSMISSION = 1u << 1u,
        BSDF_DIFFUSE = 1u << 2u,
        BSDF_GLOSSY = 1u << 3u,
        BSDF_SPECULAR = 1u << 4u,
        BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION,
    };
    struct BSDFSample {
        Vec3 wi = Vec3(0);
        Float pdf = 0.0;
        Spectrum f = Spectrum(0.0f);
        BSDFType sampled = BSDF_NONE;
    };

    struct BSDFSampleContext {
        const vec2 u1;
        const Vec3 wo;

        BSDFSampleContext(const vec2 &u1, const Vec3 &wo) : u1(u1), wo(wo) {}
    };

    class BSDFClosure {
      public:
        [[nodiscard]] virtual Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const = 0;
        [[nodiscard]] virtual Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const = 0;
        [[nodiscard]] virtual BSDFType type() const = 0;
        [[nodiscard]] bool match_flags(BSDFType flag) const { return ((uint32_t)type() & (uint32_t)flag) != 0; }
        virtual Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const = 0;
    };
    class NullBSDF : public BSDFClosure {
        [[nodiscard]] virtual Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.0; }
        [[nodiscard]] virtual Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const { return Spectrum(0.0); }
        [[nodiscard]] virtual BSDFType type() const { return BSDF_NONE; }
        virtual Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const {
            *sampledType = BSDF_NONE;
            *pdf = 0.0;
            return Spectrum(0.0);
        }
    };
    AKR_EXPORT NullBSDF *null_bsdf();
    class DiffuseBSDF : public BSDFClosure {
        Spectrum R;

      public:
        DiffuseBSDF(const Spectrum &R) : R(R) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const override {
            using namespace shader;
            if (same_hemisphere(wo, wi)) {
                return cosine_hemisphere_pdf(std::abs(cos_theta(wi)));
            }
            return 0.0f;
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const override {
            using namespace shader;
            if (same_hemisphere(wo, wi)) {
                return R * shader::InvPi;
            }
            return Spectrum(0.0f);
        }
        [[nodiscard]] BSDFType type() const override { return BSDFType(BSDF_DIFFUSE | BSDF_REFLECTION); }
        Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const override {
            using namespace shader;
            *wi = cosine_hemisphere_sampling(u);
            if (!same_hemisphere(wo, *wi)) {
                wi->y = -wi->y;
            }
            *sampledType = type();
            *pdf = cosine_hemisphere_pdf(std::abs(cos_theta(*wi)));
            return R * shader::InvPi;
        }
    };

    class MixBSDF : public BSDFClosure {
      public:
        Float fraction;
        const BSDFClosure *bsdf_A = null_bsdf();
        const BSDFClosure *bsdf_B = null_bsdf();
        MixBSDF() = default;
        MixBSDF(Float fraction, const BSDFClosure *bsdf_A, const BSDFClosure *bsdf_B)
            : fraction(fraction), bsdf_A(bsdf_A), bsdf_B(bsdf_B) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const override {
            return (1.0 - fraction) * bsdf_A->evaluate_pdf(wo, wi) + fraction * bsdf_B->evaluate_pdf(wo, wi);
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const override {
            return (1.0 - fraction) * bsdf_A->evaluate(wo, wi) + fraction * bsdf_B->evaluate(wo, wi);
        }
        [[nodiscard]] BSDFType type() const override { return BSDFType(bsdf_A->type() | bsdf_B->type()); }
        Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const override {
            Float pdf_inner = 0;
            Float pdf_select = 0;
            Spectrum f;
            if (u[0] < fraction) {
                vec2 u_(u[0] / fraction, u[1]);
                pdf_select = fraction;
                f = bsdf_A->sample(u_, wo, wi, &pdf_inner, sampledType);
            } else {
                vec2 u_((u[0] - fraction) / (1.0 - fraction), u[1]);
                pdf_select = 1.0 - fraction;
                f = bsdf_B->sample(u_, wo, wi, &pdf_inner, sampledType);
            }
            *pdf = pdf_inner * pdf_select;
            return f;
        }
    };
    class BSDF {
        const BSDFClosure *closure_;
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
        bool null() const { return closure() == nullptr; }
        void set_closure(const BSDFClosure *closure) { closure_ = closure; }
        void set_choice_pdf(Float pdf) { choice_pdf = pdf; }
        const BSDFClosure *closure() const { return closure_; }
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            auto pdf = closure()->evaluate_pdf(frame.world_to_local(wo), frame.world_to_local(wi));
            return pdf * choice_pdf;
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const {
            auto f = closure()->evaluate(frame.world_to_local(wo), frame.world_to_local(wi));
            return f;
        }

        [[nodiscard]] BSDFType type() const { return closure()->type(); }
        [[nodiscard]] bool match_flags(BSDFType flag) const { return closure()->match_flags(flag); }
        BSDFSample sample(const BSDFSampleContext &ctx) const {
            auto wo = frame.world_to_local(ctx.wo);
            Vec3 wi;
            BSDFSample sample;
            sample.f = closure()->sample(ctx.u1, wo, &wi, &sample.pdf, &sample.sampled);
            sample.wi = frame.local_to_world(wi);
            sample.pdf *= choice_pdf;
            return sample;
        }
    };
    struct MaterialEvalContext {
        Allocator<> *allocator;
        vec2 u1, u2;
        ShadingPoint sp;
        Vec3 ng, ns;
        MaterialEvalContext(Allocator<> *allocator, Sampler *sampler, const SurfaceInteraction &si)
            : MaterialEvalContext(allocator, sampler, si.texcoords, si.ng, si.ns) {}
        MaterialEvalContext(Allocator<> *allocator, Sampler *sampler, const vec2 &texcoords, const Vec3 &ng,
                            const Vec3 &ns)
            : allocator(allocator), u1(sampler->next2d()), u2(sampler->next2d()), sp(texcoords), ng(ng), ns(ns) {}
    };

    class EmissiveMaterial;
    class Material {
      public:
        virtual BSDFClosure *evaluate(MaterialEvalContext &ctx) const = 0;
        virtual bool is_emissive() const { return false; }
        virtual const EmissiveMaterial *as_emissive() const { return nullptr; }
        BSDF get_bsdf(MaterialEvalContext &ctx) const {
            auto closure = evaluate(ctx);
            BSDF bsdf(ctx.ng, ctx.ns);
            bsdf.set_closure(closure);
            return bsdf;
        }
    };
    class EmissiveMaterial : public Material {
      public:
        EmissiveMaterial(const Texture *color) : color(color) {}
        const Texture *color = nullptr;
        bool double_sided = false;
        BSDFClosure *evaluate(MaterialEvalContext &ctx) const override { return nullptr; }
        bool is_emissive() const override { return true; }
        const EmissiveMaterial *as_emissive() const override { return this; }
    };
    class MaterialNode : public SceneGraphNode {
      public:
        virtual Material *create_material(Allocator<> *) = 0;
    };

    inline std::shared_ptr<TextureNode> resolve_texture(const sdl::Value &value) {
        if (value.is_array()) {
            return create_constant_texture_rgb(load<Color3f>(value));
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

    class EmissiveMaterialNode : public MaterialNode {
      public:
        bool double_sided = false;
        std::shared_ptr<TextureNode> color;
        Material *create_material(Allocator<> *allocator) override {
            auto tex = color->create_texture(allocator);
            return allocator->new_object<EmissiveMaterial>(tex);
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = resolve_texture(value);
            }
        }
    };

    class MixMaterial : public Material {
      public:
        MixMaterial(const Texture *fraction, const Material *mat_A, const Material *mat_B)
            : fraction(fraction), mat_A(mat_A), mat_B(mat_B) {}
        const Texture *fraction = nullptr;
        const Material *mat_A = nullptr;
        const Material *mat_B = nullptr;
        BSDFClosure *evaluate(MaterialEvalContext &ctx) const override {
            auto closure = ctx.allocator->new_object<MixBSDF>(fraction->evaluate(ctx.sp)[0], mat_A->evaluate(ctx),
                                                              mat_B->evaluate(ctx));
            return closure;
        }
    };

    class MixMaterialNode final : public MaterialNode {
        std::shared_ptr<TextureNode> fraction;
        std::shared_ptr<MaterialNode> mat_A;
        std::shared_ptr<MaterialNode> mat_B;

      public:
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "fraction" || field == "frac") {
                fraction = resolve_texture(value);
            } else if (field == "first") {
                mat_A = dyn_cast<MaterialNode>(value.object());
                AKR_ASSERT_THROW(mat_A);
            } else if (field == "second") {
                mat_B = dyn_cast<MaterialNode>(value.object());
                AKR_ASSERT_THROW(mat_A);
            }
        }
        Material *create_material(Allocator<> *allocator) override {
            return allocator->new_object<MixMaterial>(fraction->create_texture(allocator),
                                                      mat_A->create_material(allocator),
                                                      mat_B->create_material(allocator));
        }
    };
} // namespace akari::render