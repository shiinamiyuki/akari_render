

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

    struct BSDFSample {
        Vec3 wi = Vec3(0);
        Float pdf = 0.0;
        Spectrum f = Spectrum(0.0f);
        BSDFType sampled = BSDFType::Unset;
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
        vec2 u1;
        ShadingPoint sp;
        Vec3 ng, ns;
        AKR_XPU MaterialEvalContext(Allocator<> *allocator, Sampler *sampler, const vec2 &texcoords, const Vec3 &ng,
                                    const Vec3 &ns)
            : allocator(allocator), u1(sampler->next2d()), sp(texcoords), ng(ng), ns(ns) {}
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
                auto closure = PassThrough(Spectrum(1));
                BSDF bsdf(ctx.ng, ctx.ns);
                bsdf.set_closure(closure);
                bsdf.set_choice_pdf(pass_through_pdf);
                return bsdf;
            }
        }
        AKR_XPU Float tr(const ShadingPoint &sp) const;
    };

    class EmissiveMaterial;
    class EmissiveMaterial {
      public:
        EmissiveMaterial(Texture color) : color(color) {}
        Texture color;
        bool double_sided = false;
        AKR_XPU BSDFClosure evaluate(MaterialEvalContext &ctx) const { return BSDFClosure(); }
        AKR_XPU Spectrum albedo(const ShadingPoint &sp) const { return color.evaluate(sp); }
        AKR_XPU Float tr(const ShadingPoint &sp) const { return color.tr(sp); }
    };
    class MaterialNode : public SceneGraphNode {
      public:
        virtual Material create_material(Allocator<> *) = 0;
    };

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

    class MixMaterial {
      public:
        MixMaterial(Texture fraction, Material mat_A, Material mat_B)
            : fraction(fraction), mat_A(mat_A), mat_B(mat_B) {}
        Texture fraction;
        Material mat_A;
        Material mat_B;
        AKR_XPU BSDFClosure evaluate(MaterialEvalContext &ctx) const {
#ifdef AKR_GPU_CODE
            // GPU can not allocate temp buffer
            astd::abort();
#else
            auto closure =
                MixBSDF(fraction.evaluate(ctx.sp)[0], ctx.allocator->new_object<BSDFClosure>(mat_A.evaluate(ctx)),
                        ctx.allocator->new_object<BSDFClosure>(mat_B.evaluate(ctx)));
            return closure;
#endif
        }
        AKR_XPU Spectrum albedo(const ShadingPoint &sp) const {
            return lerp(mat_A.albedo(sp), mat_B.albedo(sp), fraction.evaluate(sp)[0]);
        }
        AKR_XPU Float tr(const ShadingPoint &sp) const {
            return lerp(mat_A.tr(sp), mat_B.tr(sp), fraction.evaluate(sp)[0]);
        }
    };
    class DiffuseMaterial {
      public:
        DiffuseMaterial(Texture color) : color(color) {}
        Texture color;
        AKR_XPU BSDFClosure evaluate(MaterialEvalContext &ctx) const {
            auto R = color.evaluate(ctx.sp);
            return DiffuseBSDF(R);
        }
        AKR_XPU Spectrum albedo(const ShadingPoint &sp) const {
            auto R = color.evaluate(sp);
            return R;
        }
        AKR_XPU Float tr(const ShadingPoint &sp) const { return color.tr(sp); }
    };
    class GlossyMaterial {
      public:
        GlossyMaterial(Texture color, Texture roughness) : color(color), roughness(roughness) {}
        Texture color;
        Texture roughness;
        AKR_XPU BSDFClosure evaluate(MaterialEvalContext &ctx) const {
            auto R = color.evaluate(ctx.sp);
            auto r = roughness.evaluate(ctx.sp)[0];
            return MicrofacetReflection(R, r);
        }
        AKR_XPU Spectrum albedo(const ShadingPoint &sp) const {
            auto R = color.evaluate(sp);
            return R;
        }
        AKR_XPU Float tr(const ShadingPoint &sp) const { return color.tr(sp); }
    };
    AKR_XPU inline BSDFClosure Material::evaluate(MaterialEvalContext &ctx) const {
        AKR_VAR_PTR_DISPATCH(evaluate, ctx);
    }
    AKR_XPU inline Spectrum Material::albedo(const ShadingPoint &sp) const { AKR_VAR_PTR_DISPATCH(albedo, sp); }
    AKR_XPU inline Float Material::tr(const ShadingPoint &sp) const { AKR_VAR_PTR_DISPATCH(tr, sp); }
    AKR_EXPORT std::shared_ptr<MaterialNode> create_diffuse_material();
    AKR_EXPORT std::shared_ptr<MaterialNode> create_glossy_material();
    AKR_EXPORT std::shared_ptr<MaterialNode> create_emissive_material();
    AKR_EXPORT std::shared_ptr<MaterialNode> create_mix_material();
} // namespace akari::render