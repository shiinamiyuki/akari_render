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

#ifndef AKARIRENDER_BSDF_H
#define AKARIRENDER_BSDF_H

#include "interaction.h"
#include <akari/core/spectrum.h>
#include <akari/render/geometry.hpp>
#include <akari/render/sampling.hpp>
namespace akari {
    enum BSDFType : int {
        BSDF_NONE = 0u,
        BSDF_REFLECTION = 1u << 0u,
        BSDF_TRANSMISSION = 1u << 1u,
        BSDF_DIFFUSE = 1u << 2u,
        BSDF_GLOSSY = 1u << 3u,
        BSDF_SPECULAR = 1u << 4u,
        BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION,
    };

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float cos_theta(const Vector3f &w) {
        return w.y;
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float abs_cos_theta(const Vector3f &w) {
        return std::abs(cos_theta(w));
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float Cos2Theta(const Vector3f &w) {
        return w.y * w.y;
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float Sin2Theta(const Vector3f &w) {
        return 1 - Cos2Theta(w);
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float sin_theta(const Vector3f &w) {
        return std::sqrt(std::fmax(0.0f, Sin2Theta(w)));
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float Tan2Theta(const Vector3f &w) {
        return Sin2Theta(w) / Cos2Theta(w);
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float TanTheta(const Vector3f &w) {
        return std::sqrt(std::fmax(0.0f, Tan2Theta(w)));
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float cos_phi(const Vector3f &w) {
        Float sinTheta = sin_theta(w);
        return (sinTheta == 0) ? 1 : std::clamp<float>(w.x / sinTheta, -1, 1);
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float sin_phi(const Vector3f &w) {
        Float sinTheta = sin_theta(w);
        return (sinTheta == 0) ? 0 : std::clamp<float>(w.z / sinTheta, -1, 1);
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float Cos2Phi(const Vector3f &w) {
        return cos_phi(w) * cos_phi(w);
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>> inline Float Sin2Phi(const Vector3f &w) {
        return sin_phi(w) * sin_phi(w);
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>>
    inline bool same_hemisphere(const Vector3f &wo, const Vector3f &wi) {
        return wo.y * wi.y >= 0;
    }

    template <typename Vector3f, typename Float = scalar_t<Vector3f>>
    inline Vector3f reflect(const Vector3f &w, const vec3 &n) {
        return -1.0f * w + 2.0f * dot(w, n) * n;
    }
    template <typename Vector3f, typename Float = scalar_t<Vector3f>, typename Bool = replace_scalar_t<Float, bool>>
    inline Bool refract(const Vector3f &wi, const Vector3f &n, Float eta, vec3 *wt) {
        Float cosThetaI = dot(n, wi);
        Float sin2ThetaI = std::fmax(0.0f, 1.0f - cosThetaI * cosThetaI);
        Float sin2ThetaT = eta * eta * sin2ThetaI;
        if (sin2ThetaT >= 1)
            return false;

        Float cosThetaT = std::sqrt(1 - sin2ThetaT);

        *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
        return true;
    }

    template <typename Float, typename Spectrum> struct BSDFSample {
        AKR_BASIC_TYPES()
        AKR_USE_TYPES(SurfaceInteraction)
        using BSDFTypeV = replace_scalar_t<Float, BSDFType>;
        const Vector3f wo;
        Float          u0{};
        Vector2f       u{};
        Vector3f       wi{};
        Spectrum       f{};
        Float          pdf = 0;
        BSDFType       sampledType = BSDF_NONE;
        inline BSDFSample(Float u0, const Vector2f &u, const SurfaceInteraction &si);
    };

    template <typename Float, typename Spectrum> class BSDFComponent {
      public:
        AKR_BASIC_TYPES()
        explicit BSDFComponent(BSDFType type) : type(type) {}
        const BSDFType              type;
        [[nodiscard]] virtual Float evaluate_pdf(const vec3 &wo, const vec3 &wi) const {
            return abs_cos_theta(wi) * InvPi;
        }
        [[nodiscard]] virtual Spectrum evaluate(const vec3 &wo, const vec3 &wi) const = 0;
        virtual Spectrum sample(const Vector2f &u, const vec3 &wo, vec3 *wi, Float *pdf, BSDFType *sampledType) const {
            *wi = cosine_hemisphere_sampling(u);
            if (!same_hemisphere(*wi, wo)) {
                wi->y *= -1;
            }
            *pdf = abs_cos_theta(*wi) * InvPi;
            *sampledType = type;
            return evaluate(wo, *wi);
        }
        [[nodiscard]] bool is_delta() const { return ((uint32_t)type & (uint32_t)BSDF_SPECULAR) != 0; }
        [[nodiscard]] bool match_flags(BSDFType flag) const { return ((uint32_t)type & (uint32_t)flag) != 0; }
    };

    template <typename Float, typename Spectrum> class AKR_EXPORT BSDF {
        AKR_BASIC_TYPES()
        AKR_GEOMETRY_TYPES()
        AKR_USE_TYPES(BSDFComponent, BSDFSample)
        constexpr static int                       MaxBSDF = 8;
        std::array<const BSDFComponent *, MaxBSDF> components{};
        int                                        nComp = 0;
        const CoordinateSystem                     frame;
        Vector3f                                   Ng;
        Vector3f                                   Ns;

      public:
        BSDF(const Vector3f &Ng, const Vector3f &Ns) : frame(Ns), Ng(Ng), Ns(Ns) {}
        explicit BSDF(const SurfaceInteraction &si) : frame(si.Ns), Ng(si.Ng), Ns(si.Ns) {}
        void                   add_component(const BSDFComponent *comp) { components[nComp++] = comp; }
        [[nodiscard]] Float    evaluate_pdf(const Vector3f &woW, const Vector3f &wiW) const;
        [[nodiscard]] vec3     local_to_world(const Vector3f &w) const { return frame.local_to_world(w); }
        [[nodiscard]] vec3     world_to_local(const Vector3f &w) const { return frame.world_to_local(w); }
        [[nodiscard]] Spectrum evaluate(const Vector3f &woW, const Vector3f &wiW) const;
        void                   sample(BSDFSample &sample) const;
    };
    template <typename Float, typename Spectrum>
    inline BSDFSample<Float, Spectrum>::BSDFSample(Float u0, const Vector2f &u, const SurfaceInteraction &si)
        : wo(si.wo), u0(u0), u(u) {}

} // namespace akari
#endif // AKARIRENDER_BSDF_H
