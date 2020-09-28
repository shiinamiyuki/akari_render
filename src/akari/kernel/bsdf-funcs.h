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
#include <akari/common/math.h>
namespace akari {
    AKR_VARIANT struct bsdf {
        AKR_IMPORT_TYPES()
        AKR_XPU static inline Float cos_theta(const Float3 &w) { return w.y; }

        AKR_XPU static inline Float abs_cos_theta(const Float3 &w) { return std::abs(cos_theta(w)); }

        AKR_XPU static inline Float cos2_theta(const Float3 &w) { return w.y * w.y; }

        AKR_XPU static inline Float sin2_theta(const Float3 &w) { return 1 - cos2_theta(w); }

        AKR_XPU static inline Float sin_theta(const Float3 &w) { return sqrt(std::fmax(0.0f, sin2_theta(w))); }

        AKR_XPU static inline Float tan2_theta(const Float3 &w) { return sin2_theta(w) / cos2_theta(w); }

        AKR_XPU static inline Float tan_theta(const Float3 &w) { return sqrt(std::fmax(0.0f, tan2_theta(w))); }

        AKR_XPU static inline Float cos_phi(const Float3 &w) {
            Float sinTheta = sin_theta(w);
            return (sinTheta == 0) ? 1 : std::clamp<Float>(w.x / sinTheta, -1, 1);
        }
        AKR_XPU static inline Float sin_phi(const Float3 &w) {
            Float sinTheta = sin_theta(w);
            return (sinTheta == 0) ? 0 : std::clamp<Float>(w.z / sinTheta, -1, 1);
        }

        AKR_XPU static inline Float cos2_phi(const Float3 &w) { return cos_phi(w) * cos_phi(w); }
        AKR_XPU static inline Float sin2_phi(const Float3 &w) { return sin_phi(w) * sin_phi(w); }

        AKR_XPU static inline bool same_hemisphere(const Float3 &wo, const Float3 &wi) {
            return wo.y * wi.y >= 0;
        }

        AKR_XPU static inline Float3 reflect(const Float3 &w, const Float3 &n) {
            return -1.0f * w + 2.0f * dot(w, n) * n;
        }

        AKR_XPU static inline bool refract(const Float3 &wi, const Float3 &n, Float eta, Float3 *wt) {
            Float cosThetaI = dot(n, wi);
            Float sin2ThetaI = std::fmax(0.0f, 1.0f - cosThetaI * cosThetaI);
            Float sin2ThetaT = eta * eta * sin2ThetaI;
            if (sin2ThetaT >= 1)
                return false;

            Float cosThetaT = sqrt(1 - sin2ThetaT);

            *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
            return true;
        }
        AKR_XPU static inline Float fr_dielectric(Float cosThetaI, Float etaI, Float etaT) {
            bool entering = cosThetaI > 0.f;
            if (!entering) {
                std::swap(etaI, etaT);
                cosThetaI = std::abs(cosThetaI);
            }
            Float sinThetaI = sqrt(max(0.0f, 1 - cosThetaI * cosThetaI));
            Float sinThetaT = etaI / etaT * sinThetaI;
            if (sinThetaT >= 1)
                return 1;
            Float cosThetaT = sqrt(max(0.0f, 1 - sinThetaT * sinThetaT));

            Float Rpar = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
            Float Rper = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
            return 0.5 * (Rpar * Rpar + Rper * Rper);
        }

        // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
        AKR_XPU static inline Spectrum fr_conductor(Float cosThetaI, const Spectrum &etaI, const Spectrum &etaT,
                                                    const Spectrum &k) {
            Float CosTheta2 = cosThetaI * cosThetaI;
            Float SinTheta2 = 1 - CosTheta2;
            Spectrum Eta = etaT / etaI;
            Spectrum Etak = k / etaI;
            Spectrum Eta2 = Eta * Eta;
            Spectrum Etak2 = Etak * Etak;

            Spectrum t0 = Eta2 - Etak2 - SinTheta2;
            Spectrum a2plusb2 = sqrt(t0 * t0 + 4.0f * Eta2 * Etak2);
            Spectrum t1 = a2plusb2 + CosTheta2;
            Spectrum a = sqrt(0.5f * (a2plusb2 + t0));
            Spectrum t2 = 2.0f * a * cosThetaI;
            Spectrum Rs = (t1 - t2) / (t1 + t2);

            Spectrum t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2;
            Spectrum t4 = t2 * SinTheta2;
            Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);

            return 0.5 * (Rp + Rs);
        }
    };
} // namespace akari