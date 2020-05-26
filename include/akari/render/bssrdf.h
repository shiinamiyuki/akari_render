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

#ifndef AKARIRENDER_BSSRDF_H
#define AKARIRENDER_BSSRDF_H
#include <akari/render/reflection.h>
namespace akari {
    class BSSRDF {
      protected:
        const SurfaceInteraction &po;
        Float eta;

      public:
        BSSRDF(const SurfaceInteraction &po, Float eta) : po(po), eta(eta) {}
        virtual Spectrum S(const SurfaceInteraction &pi, const Vector3f &wi) = 0;
        virtual Spectrum sample_S(const Scene &scene, Float u1, const Point2f &u2, MemoryArena &arena,
                                  SurfaceInteraction *si, Float *pdf) const = 0;
    };
    // ???
    inline Float fresnel_moment1(Float eta) {
        Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
        if (eta < 1)
            return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 + 2.49277f * eta4 - 0.68441f * eta5;
        else
            return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 - 1.27198f * eta4 + 0.12746f * eta5;
    }

    inline Float fresnel_moment2(Float eta) {
        Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
        if (eta < 1) {
            return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 + 0.07883f * eta4 + 0.04860f * eta5;
        } else {
            Float r_eta = 1 / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
            return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 + 458.843f * r_eta + 404.557f * eta -
                   189.519f * eta2 + 54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5;
        }
    }
    class SeparableBSSRDF;
    class Material;
    class AKR_EXPORT SeparableBSSRDF : public BSSRDF {
        vec3 Ns;
        Frame3f local_frame;
        const Material *material;
        friend class SeparableBSSRDFAdapter;
        const TransportMode mode;
      public:
        SeparableBSSRDF(const SurfaceInteraction &po, Float eta, const Material *material, TransportMode mode)
            : BSSRDF(po, eta), Ns(po.Ns), local_frame(po.Ns), material(material), mode(mode) {}
        virtual Spectrum Sr(Float d) const = 0;
        Spectrum Sp(const SurfaceInteraction &pi) const { return Sr(length(po.p - pi.p)); }
        Spectrum Sw(const Vector3f &w) const {
            Float c = 1 - 2 * fresnel_moment1(1 / eta);
            return (1 - fr_dielectric(cos_theta(w), 1, eta)) / (c * Pi);
        }
        Spectrum S(const SurfaceInteraction &pi, const Vector3f &wi) override {
            return (1.0f - fr_dielectric(dot(po.wo, Ns), 1.0f, eta)) * Sp(pi) * Sw(w);
        }
        Spectrum sample_S(const Scene &scene, Float u1, const Point2f &u2, MemoryArena &arena, SurfaceInteraction *si,
                          Float *pdf) const;
        Spectrum sample_Sp(const Scene &scene, Float u1, const Point2f &u2, MemoryArena &arena, SurfaceInteraction *si,
                           Float *pdf) const;
    };

    class SeparableBSSRDFAdapter : public BSDFComponent {
        const SeparableBSSRDF *bssrdf;
      public:
        SeparableBSSRDFAdapter(const SeparableBSSRDF *bssrdf)
            : BSDFComponent(BSDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), bssrdf(bssrdf) {}
        [[nodiscard]] Spectrum evaluate(const vec3 &wo, const vec3 &wi) const {
            Spectrum f = bssrdf->Sw(wi);
            if (bssrdf->mode == TransportMode::ERadiance)
                f *= bssrdf->eta * bssrdf->eta;

            return f;
        }
    };
} // namespace akari

#endif