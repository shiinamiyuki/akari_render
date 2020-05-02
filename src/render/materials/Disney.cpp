// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#include <akari/core/plugin.h>
#include <akari/render/material.h>
#include <akari/render/reflection.h>

namespace akari {
    inline Float SchlickWeight(Float cosTheta) { return Power<5>(std::clamp<float>(1 - cosTheta, 0, 1)); }
    inline Float FrSchlick(Float R0, Float cosTheta) { return lerp(SchlickWeight(cosTheta), R0, 1.0f); }
    inline Spectrum FrSchlick(const Spectrum &R0, Float cosTheta) {
        return lerp(Spectrum(SchlickWeight(cosTheta)), R0, Spectrum(1));
    }
    class DisneyDiffuse : public BSDFComponent {
        Spectrum R;

      public:
        DisneyDiffuse(const Spectrum &R) : BSDFComponent(BSDFType(BSDF_DIFFUSE | BSDF_REFLECTION)), R(R) {}
        [[nodiscard]] Spectrum evaluate(const vec3 &wo, const vec3 &wi) const override {
            Float Fo = SchlickWeight(abs_cos_theta(wo));
            Float Fi = SchlickWeight(abs_cos_theta(wi));
            return R * InvPi * (1 - Fo * 0.5) * (1 - Fi * 0.5);
        }
    };

    inline Spectrum CalculateTint(const Spectrum &baseColor) {
        auto luminance = baseColor.luminance();
        return luminance > 0 ? baseColor / luminance : Spectrum(1);
    }

    class DisneySheen : public BSDFComponent {
        const Spectrum R;
        const Spectrum sheen;
        const Spectrum sheenTint;

      public:
        DisneySheen(const Spectrum &R, const Spectrum &sheen, const Spectrum &sheenTint)
            : BSDFComponent(BSDFType(BSDF_GLOSSY | BSDF_REFLECTION)), R(R), sheen(sheen), sheenTint(sheenTint) {}
        [[nodiscard]] Spectrum evaluate(const vec3 &wo, const vec3 &wi) const override {
            auto wh = wi + wo;
            if (all(equal(wh, vec3(0)))) {
                return Spectrum(0);
            }
            wh = normalize(wh);
            Float d = dot(wh, wi);
            return sheen * lerp(Spectrum(1), CalculateTint(R), sheenTint) * SchlickWeight(d);
        }
    };

    static inline Float D_GTR1(Float cosThetaH, Float alpha) {
        if (alpha >= 1)
            return InvPi;
        auto a2 = alpha * alpha;
        return (a2 - 1) / (Pi * log(a2)) * (1.0f / (1 + (a2 - 1) * cosThetaH * cosThetaH));
    }
    static inline Float SmithGGX_G1(Float cosTheta, Float alpha) {
        auto a2 = alpha * alpha;
        return 1.0 / (cosTheta + std::sqrt(a2 + cosTheta - a2 * cosTheta * cosTheta));
    }
    class DisneyClearCoat : public BSDFComponent {
        Float clearcoat;
        Float alpha;

      public:
        DisneyClearCoat(Float clearcoat, Float alpha)
            : BSDFComponent(BSDFType(BSDF_GLOSSY | BSDF_REFLECTION)), clearcoat(clearcoat), alpha(alpha) {}
        [[nodiscard]] Spectrum evaluate(const vec3 &wo, const vec3 &wi) const override {
            auto wh = wi + wo;
            if (all(equal(wh, vec3(0)))) {
                return Spectrum(0);
            }
            wh = normalize(wh);
            Float D = D_GTR1(abs_cos_theta(wh), alpha);
            Float F = FrSchlick(0.04, dot(wo, wh));
            Float G = SmithGGX_G1(abs_cos_theta(wi), 0.25) * SmithGGX_G1(abs_cos_theta(wo), 0.25);
            return Spectrum(0.25 * clearcoat * D * F * G);
        }
        [[nodiscard]] Float evaluate_pdf(const vec3 &wo, const vec3 &wi) const override {
            if (!same_hemisphere(wo, wi))
                return 0;
            auto wh = wi + wo;
            if (wh.x == 0 && wh.y == 0 && wh.z == 0)
                return 0;
            wh = normalize(wh);
            return D_GTR1(abs_cos_theta(wh), alpha) * abs_cos_theta(wh) / (4.0 * dot(wh, wo));
        }
        Spectrum sample(const vec2 &u, const vec3 &wo, vec3 *wi, Float *pdf, BSDFType *sampledType) const override {
            auto a2 = alpha * alpha;
            auto cosTheta = std::sqrt(std::fmax(0.0f, 1 - (std::pow(a2, 1 - u[0])) / (1 - a2)));
            auto sinTheta = std::sqrt(std::fmax(0.0f, 1 - cosTheta * cosTheta));
            auto phi = 2.0f * Pi * u[1];
            auto wh = vec3(std::cos(phi) * sinTheta, cosTheta, std::sin(phi) * sinTheta);
            if (!same_hemisphere(wo, wh))
                wh = -wh;
            *wi = reflect(wo, wh);
            *pdf = evaluate_pdf(wo, *wi);
            return evaluate(wo, *wi);
        }
    };

    class DisneyFresnel : public Fresnel {
        const Spectrum R0;
        const Float metallic, eta;

      public:
        DisneyFresnel(const Spectrum &R0, Float metallic, Float eta) : R0(R0), metallic(metallic), eta(eta) {}
        [[nodiscard]] Spectrum evaluate(Float cosThetaI) const override {
            return lerp(Spectrum(fr_dielectric(cosThetaI, 1.0f, eta)), FrSchlick(R0, cosThetaI), Spectrum(metallic));
        }
    };

    class DisneyMaterial : public Material {
        std::shared_ptr<Texture> baseColor, subsurface, metallic, specular, specularTint, roughness, anisotropic, sheen,
            sheenTint, clearcoat, clearcoatGlass, ior, specTrans;

      public:
        AKR_SER(baseColor, subsurface, metallic, specular, specularTint, roughness, anisotropic, sheen, sheenTint,
                clearcoat, clearcoatGlass, ior, specTrans)
        AKR_DECL_COMP(DisneyMaterial, "DisneyMaterial")
        void compute_scattering_functions(SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,
                                          Float scale) const override {
            si->bsdf = arena.alloc<BSDF>(*si);
//            Spectrum color = baseColor->evaluate(si->sp);
//            Float metallicWeight = metallic->evaluate(si->sp)[0];
//            Float eta = ior->evaluate(si->sp)[0];
//            Float trans = specTrans->evaluate(si->sp)[0];
//            Float diffuseWeight = (1.0f - trans) * (1.0f - metallicWeight);
//            Float transWeight = trans * (1.0f - metallicWeight);
//            Float alpha = roughness->evaluate(si->sp)[0];
//            if(diffuseWeight > 0){
//
//            }
        }
        bool support_bidirectional() const override { return true; }
        void commit() override {
            if (baseColor)
                baseColor->commit();
            if (subsurface)
                subsurface->commit();
            if (metallic)
                metallic->commit();
            if (specular)
                specular->commit();
            if (specularTint)
                specularTint->commit();
            if (roughness)
                roughness->commit();
            if (anisotropic)
                anisotropic->commit();
            if (sheen)
                sheen->commit();
            if (sheenTint)
                sheenTint->commit();
            if (clearcoat)
                clearcoat->commit();
            if (clearcoatGlass)
                clearcoatGlass->commit();
            if(ior)
                ior->commit();
            if(specTrans)
                specTrans->commit();
        }
    };
    AKR_EXPORT_COMP(DisneyMaterial, "Material")
} // namespace akari
