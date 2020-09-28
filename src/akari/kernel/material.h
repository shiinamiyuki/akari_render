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

#include <akari/common/variant.h>
#include <akari/common/math.h>
#include <akari/kernel/bsdf-funcs.h>
#include <akari/kernel/microfacet.h>
#include <akari/kernel/texture.h>
#include <akari/kernel/sampler.h>
#include <akari/kernel/interaction.h>

namespace akari {
    enum BSDFType : int {
        BSDF_NONE = 0u,
        BSDF_REFLECTION = 1u << 0u,
        BSDF_TRANSMISSION = 1u << 1u,
        BSDF_DIFFUSE = 1u << 2u,
        BSDF_GLOSSY = 1u << 3u,
        // BSDF_SPECULAR = 1u << 4u,
        BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION,
    };

    AKR_VARIANT struct BSDFSample {
        AKR_IMPORT_TYPES()
        Float3 wi = Float3(0);
        Float pdf = 0.0;
        Spectrum f = Spectrum(0.0f);
        BSDFType sampled = BSDF_NONE;
    };
    AKR_VARIANT struct BSDFSampleContext {
        AKR_IMPORT_TYPES()
        const float2 u1;
        const Float3 wo;
        AKR_XPU BSDFSampleContext(const float2 &u1, const Float3 &wo) : u1(u1), wo(wo) {}
    };
    AKR_VARIANT class DiffuseBSDF {
        AKR_IMPORT_TYPES();
        Spectrum R;

      public:
        AKR_XPU DiffuseBSDF(const Spectrum &R) : R(R) {}
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Float3 &wo, const Float3 &wi) const {
            if (bsdf<C>::same_hemisphere(wo, wi)) {
                return sampling<C>::cosine_hemisphere_pdf(std::abs(bsdf<C>::cos_theta(wi)));
            }
            return 0.0f;
        }
        [[nodiscard]] AKR_XPU Spectrum evaluate(const Float3 &wo, const Float3 &wi) const {
            if (bsdf<C>::same_hemisphere(wo, wi)) {
                return R * Constants<Float>::InvPi();
            }
            return Spectrum(0.0f);
        }
        [[nodiscard]] AKR_XPU BSDFType type() const { return BSDFType(BSDF_DIFFUSE | BSDF_REFLECTION); }
        AKR_XPU Spectrum sample(const float2 &u, const Float3 &wo, Float3 *wi, Float *pdf,
                                BSDFType *sampledType) const {
            *wi = sampling<C>::cosine_hemisphere_sampling(u);
            if (!bsdf<C>::same_hemisphere(wo, *wi)) {
                wi->y = -wi->y;
            }
            *sampledType = type();
            *pdf = sampling<C>::cosine_hemisphere_pdf(std::abs(bsdf<C>::cos_theta(*wi)));
            return R * Constants<Float>::InvPi();
        }
    };
    AKR_VARIANT class MicrofacetReflection {
        AKR_IMPORT_TYPES();
        Spectrum R;
        Float roughness;
        typename microfacet<C>::MicrofacetModel model;

      public:
        AKR_XPU MicrofacetReflection(const Spectrum &R, Float roughness)
            : R(R), roughness(roughness), model(microfacet<C>::MicrofacetType::EGGX, roughness) {}
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Float3 &wo, const Float3 &wi) const {
            if (bsdf<C>::same_hemisphere(wo, wi)) {
                auto wh = normalize(wo + wi);
                return model.evaluate_pdf(wh) / (Float(4.0f) * dot(wo, wh));
            }
            return 0.0f;
        }
        [[nodiscard]] AKR_XPU Spectrum evaluate(const Float3 &wo, const Float3 &wi) const {
            if (bsdf<C>::same_hemisphere(wo, wi)) {
                Float cosThetaO = bsdf<C>::abs_cos_theta(wo);
                Float cosThetaI = bsdf<C>::abs_cos_theta(wi);
                auto wh = (wo + wi);
                if (cosThetaI == 0 || cosThetaO == 0)
                    return Spectrum(0);
                if (wh.x == 0 && wh.y == 0 && wh.z == 0)
                    return Spectrum(0);
                wh = normalize(wh);
                if (wh.y < 0) {
                    wh = -wh;
                }
                auto F = 1.0f; // fresnel->evaluate(dot(wi, wh));
                return R * (model.D(wh) * model.G(wo, wi, wh) * F / (Float(4.0f) * cosThetaI * cosThetaO));
            }
            return Spectrum(0.0f);
        }
        [[nodiscard]] AKR_XPU BSDFType type() const { return BSDFType(BSDF_GLOSSY | BSDF_REFLECTION); }
        AKR_XPU Spectrum sample(const float2 &u, const Float3 &wo, Float3 *wi, Float *pdf,
                                BSDFType *sampledType) const {
            *sampledType = type();
            auto wh = model.sample_wh(wo, u);
            *wi = bsdf<C>::reflect(wo, wh);
            if (!bsdf<C>::same_hemisphere(wo, *wi)) {
                *pdf = 0;
                return Spectrum(0);
            } else {
                if (wh.y < 0) {
                    wh = -wh;
                }
                *pdf = model.evaluate_pdf(wh) / (Float(4.0f) * abs(dot(wo, wh)));
            }
            return evaluate(wo, *wi);
        }
    };
    AKR_VARIANT class BSDFClosure : Variant<DiffuseBSDF<C>, MicrofacetReflection<C>> {
      public:
        using Variant<DiffuseBSDF<C>, MicrofacetReflection<C>>::Variant;
        AKR_IMPORT_TYPES();
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Float3 &wo, const Float3 &wi) const {
            AKR_VAR_DISPATCH(evaluate_pdf, wo, wi);
        }
        [[nodiscard]] AKR_XPU Spectrum evaluate(const Float3 &wo, const Float3 &wi) const {
            AKR_VAR_DISPATCH(evaluate, wo, wi);
        }
        [[nodiscard]] AKR_XPU BSDFType type() const { AKR_VAR_DISPATCH(type); }
        [[nodiscard]] AKR_XPU bool match_flags(BSDFType flag) const { return ((uint32_t)type() & (uint32_t)flag) != 0; }
        AKR_XPU Spectrum sample(const float2 &u, const Float3 &wo, Float3 *wi, Float *pdf,
                                BSDFType *sampledType) const {
            AKR_VAR_DISPATCH(sample, u, wo, wi, pdf, sampledType);
        }
    };

    AKR_VARIANT class BSDF {
        AKR_IMPORT_TYPES()
        BSDFClosure<C> closure_;
        Float3 ng, ns;
        Frame3f frame;
        Float choice_pdf = 1.0f;

      public:
        BSDF() = default;
        AKR_XPU explicit BSDF(const Float3 &ng, const Float3 &ns) : ng(ng), ns(ns) { frame = Frame3f(ns); }
        AKR_XPU bool null() const { return closure().null(); }
        AKR_XPU void set_closure(const BSDFClosure<C> &closure) { closure_ = closure; }
        AKR_XPU void set_choice_pdf(Float pdf) { choice_pdf = pdf; }
        AKR_XPU const BSDFClosure<C> &closure() const { return closure_; }
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Float3 &wo, const Float3 &wi) const {
            auto pdf = closure().evaluate_pdf(frame.world_to_local(wo), frame.world_to_local(wi));
            return pdf * choice_pdf;
        }
        [[nodiscard]] AKR_XPU Spectrum evaluate(const Float3 &wo, const Float3 &wi) const {
            auto f = closure().evaluate(frame.world_to_local(wo), frame.world_to_local(wi));
            return f;
        }

        [[nodiscard]] AKR_XPU BSDFType type() const { return closure().type(); }
        [[nodiscard]] AKR_XPU bool match_flags(BSDFType flag) const { return closure().match_flags(flag); }
        AKR_XPU BSDFSample<C> sample(const BSDFSampleContext<C> &ctx) const {
            auto wo = frame.world_to_local(ctx.wo);
            Float3 wi;
            BSDFSample<C> sample;
            sample.f = closure().sample(ctx.u1, wo, &wi, &sample.pdf, &sample.sampled);
            sample.wi = frame.local_to_world(wi);
            sample.pdf *= choice_pdf;
            return sample;
        }
    };
    AKR_VARIANT struct MaterialEvalContext {
        AKR_IMPORT_TYPES()
        float2 u1, u2;
        float2 texcoords;
        Float3 ng, ns;
        MaterialEvalContext() = default;
        AKR_XPU MaterialEvalContext(Sampler<C> sampler, const SurfaceInteraction<C> &si)
            : MaterialEvalContext(sampler, si.texcoords, si.ng, si.ns) {}
        AKR_XPU MaterialEvalContext(Sampler<C> sampler, const float2 &texcoords, const Float3 &ng,
                                    const Float3 &ns)
            : u1(sampler.next2d()), u2(sampler.next2d()), texcoords(texcoords), ng(ng), ns(ns) {}
    };

    AKR_VARIANT class DiffuseMaterial {
      public:
        DiffuseMaterial(Texture<C> *color) : color(color) {}
        AKR_IMPORT_TYPES()
        Texture<C> *color;
        AKR_XPU BSDF<C> get_bsdf(MaterialEvalContext<C> &ctx) const {
            auto R = color->evaluate(ctx.texcoords);
            BSDF<C> bsdf(ctx.ng, ctx.ns);
            bsdf.set_closure((DiffuseBSDF<C>(R)));
            return bsdf;
        }
    };
    AKR_VARIANT class GlossyMaterial {
      public:
        AKR_IMPORT_TYPES()
        const Texture<C> *color = nullptr;
        const Texture<C> *roughness = nullptr;
        GlossyMaterial(Texture<C> *color, const Texture<C> *roughness) : color(color), roughness(roughness) {}
        AKR_XPU BSDF<C> get_bsdf(MaterialEvalContext<C> &ctx) const {
            auto R = color->evaluate(ctx.texcoords);
            auto roughness_ = roughness->evaluate(ctx.texcoords).x;
            roughness_ *= roughness_;
            BSDF<C> bsdf(ctx.ng, ctx.ns);
            bsdf.set_closure(MicrofacetReflection<C>(R, roughness_));
            return bsdf;
        }
    };
    AKR_VARIANT class EmissiveMaterial {
      public:
        AKR_IMPORT_TYPES()
        const Texture<C> *color;
        bool double_sided = false;
        AKR_XPU EmissiveMaterial(const Texture<C> *color, bool double_sided = false)
            : color(color), double_sided(double_sided) {}
    };
    AKR_VARIANT class MixMaterial;
    AKR_VARIANT class MixMaterial {
      public:
        AKR_IMPORT_TYPES()
        const Texture<C> *fraction;
        const Material<C> *material_A, *material_B;
        MixMaterial(const Texture<C> *fraction, const Material<C> *material_A, const Material<C> *material_B)
            : fraction(fraction), material_A(material_A), material_B(material_B) {}
    };
    AKR_VARIANT class Material
        : public Variant<DiffuseMaterial<C>, GlossyMaterial<C>, EmissiveMaterial<C>, MixMaterial<C>> {
      public:
        AKR_IMPORT_TYPES()
        using Variant<DiffuseMaterial<C>, GlossyMaterial<C>, EmissiveMaterial<C>, MixMaterial<C>>::Variant;

        AKR_XPU astd::pair<const Material<C> *, Float> select_material(Float &u, const float2 &texcoords) const {
            Float choice_pdf = 1.0f;
            auto ptr = this;
            while (ptr->template isa<MixMaterial<C>>()) {
                auto frac = ptr->template get<MixMaterial<C>>()->fraction->evaluate(texcoords).x;
                if (u < frac) {
                    u = u / frac;
                    ptr = ptr->template get<MixMaterial<C>>()->material_B;
                    choice_pdf *= 1.0f / frac;
                } else {
                    u = (u - frac) / (1.0f - frac);
                    ptr = ptr->template get<MixMaterial<C>>()->material_A;
                    choice_pdf *= 1.0f / (1.0f - frac);
                }
            }
            return {ptr, choice_pdf};
        }

      private:
        AKR_XPU BSDF<C> get_bsdf0(MaterialEvalContext<C> &ctx) const {
            return this->dispatch([&](auto &&arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MixMaterial<C>> || std::is_same_v<T, EmissiveMaterial<C>>) {
                    return BSDF<C>();
                } else
                    return arg.get_bsdf(ctx);
            });
        }

      public:
        AKR_XPU static BSDF<C> get_bsdf(const astd::pair<const Material<C> *, Float> &pair,
                                        MaterialEvalContext<C> &ctx) {
            auto [mat, choice_pdf] = pair;
            auto bsdf = mat->get_bsdf0(ctx);
            bsdf.set_choice_pdf(choice_pdf);
            return bsdf;
        }
        AKR_XPU BSDF<C> get_bsdf(MaterialEvalContext<C> &ctx) const {
            auto [mat, choice_pdf] = select_material(ctx.u1[0], ctx.texcoords);
            auto bsdf = mat->get_bsdf0(ctx);
            bsdf.set_choice_pdf(choice_pdf);
            return bsdf;
        }
    };

} // namespace akari