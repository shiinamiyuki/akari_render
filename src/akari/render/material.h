

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
#include <akari/render/interaction.h>
#include <akari/render/texture.h>
#include <akari/render/closure.h>
#include <optional>

namespace akari::render {
    class Material;
    class DiffuseMaterial;
    class GlossyMaterial;
    class EmissiveMaterial;
    class MixMaterial;

    struct BSDFSample {
        Vec3 wi = Vec3(0);
        Float pdf = 0.0;
        Spectrum f = Spectrum(0.0f);
        BSDFType sampled = BSDF_NONE;
    };

    struct BSDFSampleContext {
        const vec2 u1;
        const Vec3 wo;

        AKR_XPU BSDFSampleContext(const vec2 &u1, const Vec3 &wo) : u1(u1), wo(wo) {}
    };

    class BSDF {
        BSDFClosure closure_;
        Vec3 ng, ns;
        Frame frame;
        Float choice_pdf = 1.0f;

      public:
        BSDF() = default;
        AKR_XPU explicit BSDF(const Vec3 &ng, const Vec3 &ns) : ng(ng), ns(ns) {
            frame = Frame(ns);
            (void)this->ng;
            (void)this->ns;
        }
        AKR_XPU bool null() const { return closure().null(); }
        AKR_XPU void set_closure(const BSDFClosure &closure) { closure_ = closure; }
        AKR_XPU void set_choice_pdf(Float pdf) { choice_pdf = pdf; }
        AKR_XPU const BSDFClosure &closure() const { return closure_; }
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            auto pdf = closure().evaluate_pdf(frame.world_to_local(wo), frame.world_to_local(wi));
            return pdf * choice_pdf;
        }
        [[nodiscard]] AKR_XPU Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const {
            auto f = closure().evaluate(frame.world_to_local(wo), frame.world_to_local(wi));
            return f;
        }

        [[nodiscard]] AKR_XPU BSDFType type() const { return closure().type(); }
        [[nodiscard]] AKR_XPU bool match_flags(BSDFType flag) const { return closure().match_flags(flag); }
        AKR_XPU BSDFSample sample(const BSDFSampleContext &ctx) const {
            auto wo = frame.world_to_local(ctx.wo);
            Vec3 wi;
            BSDFSample sample;
            sample.f = closure().sample(ctx.u1, wo, &wi, &sample.pdf, &sample.sampled);
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
        AKR_XPU MaterialEvalContext(Allocator<> *allocator, Sampler *sampler, const SurfaceInteraction &si)
            : MaterialEvalContext(allocator, sampler, si.texcoords, si.ng, si.ns) {}
        AKR_XPU MaterialEvalContext(Allocator<> *allocator, Sampler *sampler, const vec2 &texcoords, const Vec3 &ng,
                                    const Vec3 &ns)
            : allocator(allocator), u1(sampler->next2d()), u2(sampler->next2d()), sp(texcoords), ng(ng), ns(ns) {}
    };
    class Material : public Variant<const DiffuseMaterial *, const GlossyMaterial *, const EmissiveMaterial *,
                                    const MixMaterial *> {
      public:
        using Variant::Variant;
        AKR_XPU inline BSDFClosure evaluate(MaterialEvalContext &ctx) const;
        AKR_XPU inline Spectrum albedo(const ShadingPoint &sp) const;
        AKR_XPU bool is_emissive() const { return this->isa<const EmissiveMaterial *>(); }
        AKR_XPU const EmissiveMaterial *as_emissive() const { return *this->get<const EmissiveMaterial *>(); }
        AKR_XPU bool is_mix() const { return this->isa<const MixMaterial *>(); }
        AKR_XPU const MixMaterial *as_mix() const { return *this->get<const MixMaterial *>(); }
        AKR_XPU BSDF get_bsdf(MaterialEvalContext &ctx) const {
            auto closure = evaluate(ctx);
            BSDF bsdf(ctx.ng, ctx.ns);
            bsdf.set_closure(closure);
            return bsdf;
        }
    };

    class EmissiveMaterial;
    class EmissiveMaterial {
      public:
        EmissiveMaterial(const Texture *color) : color(color) {}
        const Texture *color = nullptr;
        bool double_sided = false;
        BSDFClosure evaluate(MaterialEvalContext &ctx) const { return BSDFClosure(); }
        Spectrum albedo(const ShadingPoint &sp) const { return color->evaluate(sp); }
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

    class MixMaterial {
      public:
        MixMaterial(const Texture *fraction, const Material *mat_A, const Material *mat_B)
            : fraction(fraction), mat_A(mat_A), mat_B(mat_B) {}
        const Texture *fraction = nullptr;
        const Material *mat_A = nullptr;
        const Material *mat_B = nullptr;
        AKR_EXPORT BSDFClosure evaluate(MaterialEvalContext &ctx) const {
            auto closure =
                MixBSDF(fraction->evaluate(ctx.sp)[0], ctx.allocator->new_object<BSDFClosure>(mat_A->evaluate(ctx)),
                        ctx.allocator->new_object<BSDFClosure>(mat_B->evaluate(ctx)));
            return closure;
        }
        AKR_EXPORT Spectrum albedo(const ShadingPoint &sp) const {
            return lerp(mat_A->albedo(sp), mat_B->albedo(sp), fraction->evaluate(sp)[0]);
        }
    };
    class DiffuseMaterial {
      public:
        DiffuseMaterial(const Texture *color) : color(color) {}
        const Texture *color;
        BSDFClosure evaluate(MaterialEvalContext &ctx) const {
            auto R = color->evaluate(ctx.sp);
            return DiffuseBSDF(R);
        }
        Spectrum albedo(const ShadingPoint &sp) const {
            auto R = color->evaluate(sp);
            return R;
        }
    };
    class GlossyMaterial {
      public:
        GlossyMaterial(const Texture *color, const Texture *roughness) : color(color), roughness(roughness) {}
        const Texture *color;
        const Texture *roughness;
        BSDFClosure evaluate(MaterialEvalContext &ctx) const {
            auto R = color->evaluate(ctx.sp);
            auto r = roughness->evaluate(ctx.sp)[0];
            return MicrofacetReflection(R, r);
        }
        Spectrum albedo(const ShadingPoint &sp) const {
            auto R = color->evaluate(sp);
            return R;
        }
    };
    AKR_XPU inline BSDFClosure Material::evaluate(MaterialEvalContext &ctx) const {
        AKR_VAR_PTR_DISPATCH(evaluate, ctx);
    }
    AKR_XPU inline Spectrum Material::albedo(const ShadingPoint &sp) const { AKR_VAR_PTR_DISPATCH(albedo, sp); }
    AKR_EXPORT std::shared_ptr<MaterialNode> create_diffuse_material();
    AKR_EXPORT std::shared_ptr<MaterialNode> create_glossy_material();
    AKR_EXPORT std::shared_ptr<MaterialNode> create_emissive_material();
    AKR_EXPORT std::shared_ptr<MaterialNode> create_mix_material();
} // namespace akari::render