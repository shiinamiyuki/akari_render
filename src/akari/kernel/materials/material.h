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

#include <akari/common/taggedpointer.h>
#include <akari/common/math.h>

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
    AKR_VARIANT struct bsdf {
        AKR_IMPORT_TYPES()
        static inline Float cos_theta(const Vector3f &w) { return w.y(); }

        static inline Float abs_cos_theta(const Vector3f &w) { return std::abs(cos_theta(w)); }

        static inline Float cos2_theta(const Vector3f &w) { return w.y() * w.y(); }

        static inline Float sin2_theta(const Vector3f &w) { return 1 - cos2_theta(w); }

        static inline Float sin_theta(const Vector3f &w) { return std::sqrt(std::fmax(0.0f, sin2_theta(w))); }

        static inline Float tan2_theta(const Vector3f &w) { return sin2_theta(w) / cos2_theta(w); }

        static inline Float tan_theta(const Vector3f &w) { return std::sqrt(std::fmax(0.0f, tan2_theta(w))); }

        static inline Float cos_phi(const Vector3f &w) {
            Float sinTheta = sin_theta(w);
            return (sinTheta == 0) ? 1 : std::clamp<Float>(w.x / sinTheta, -1, 1);
        }
        static inline Float sin_phi(const Vector3f &w) {
            Float sinTheta = sin_theta(w);
            return (sinTheta == 0) ? 0 : std::clamp<Float>(w.z / sinTheta, -1, 1);
        }

        static inline Float cos2_phi(const Vector3f &w) { return cos_phi(w) * cos_phi(w); }
        static inline Float sin2_phi(const Vector3f &w) { return sin_phi(w) * sin_phi(w); }

        static inline bool same_hemisphere(const Vector3f &wo, const Vector3f &wi) { return wo.y() * wi.y() >= 0; }

        static inline Vector3f reflect(const Vector3f &w, const Normal3f &n) {
            return -1.0f * w + 2.0f * dot(w, n) * n;
        }

        static inline bool refract(const Vector3f &wi, const Normal3f &n, Float eta, Vector3f *wt) {
            Float cosThetaI = dot(n, wi);
            Float sin2ThetaI = std::fmax(0.0f, 1.0f - cosThetaI * cosThetaI);
            Float sin2ThetaT = eta * eta * sin2ThetaI;
            if (sin2ThetaT >= 1)
                return false;

            Float cosThetaT = std::sqrt(1 - sin2ThetaT);

            *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
            return true;
        }
    };

    AKR_VARIANT struct BSDFSample {
        AKR_IMPORT_TYPES()
        const Point2f u1;
        const Vector3f wo;
        Vector3f wi = Vector3f(0);
        Float pdf = 0.0;
        Spectrum f = Spectrum(0.0f);
        BSDFType sampled = BSDF_NONE;
    };
    AKR_VARIANT struct BSDFSampleContext {
        AKR_IMPORT_TYPES()
        Vector3f wi;
        Normal3f ng, ns;
        Point3f p;
    };

    AKR_VARIANT class BSDFClosure {
      public:
        AKR_IMPORT_TYPES();
        [[nodiscard]] Float evaluate_pdf(const Vector3f &wo, const Vector3f &wi) const;
        [[nodiscard]] Spectrum evaluate(const Vector3f &wo, const Vector3f &wi) const;
        [[nodiscard]] BSDFType type() const;
        [[nodiscard]] bool is_delta() const { return ((uint32_t)type() & (uint32_t)BSDF_SPECULAR) != 0; }
        [[nodiscard]] bool match_flags(BSDFType flag) const { return ((uint32_t)type() & (uint32_t)flag) != 0; }
        Spectrum sample(const Point2f &u, const Vector3f &wo, Vector3f *wi, Float *pdf, BSDFType *sampledType);
    };

    AKR_VARIANT class BSDF {
        AKR_IMPORT_TYPES()
        BSDFClosure<C> closure_ = nullptr;
        Normal3f ng, ns;
        Frame3f frame;

      public:
        explicit BSDF(const Normal3f &ng, const Normal3f &ns) : ng(ng), ns(ns) { frame = Frame3f(ns); }
        void set_closure(BSDFClosure<C> closure) { closure_ = closure; }
        [[nodiscard]] BSDFClosure<C> closure() const { return closure_; }
        [[nodiscard]] Float evaluate_pdf(const Vector3f &wo, const Vector3f &wi) const {
            auto pdf = closure_.evaluate_pdf(frame.world_to_local(wo), frame.world_to_local(wi));
            return pdf;
        }
        [[nodiscard]] Spectrum evaluate(const Vector3f &wo, const Vector3f &wi) const {
            auto f = closure_.evaluate(frame.world_to_local(wo), frame.world_to_local(wi));
            return f;
        }

        [[nodiscard]] BSDFType type() const { return closure_.type(); }
        [[nodiscard]] bool is_delta() const { return closure_.is_delta(); }
        [[nodiscard]] bool match_flags(BSDFType flag) const { closure_.match_flags(flag); }
        void sample(BSDFSample<C> *sample) {
            auto wo = frame.world_to_local(sample->wo);
            Vector3f wi;
            sample->f = closure().sample(sample->u1, wo, &wi, sample->pdf, sample->sampled);
            sample->wi = frame.local_to_world(wi);
        }
    };

    AKR_VARIANT class Material {};

} // namespace akari