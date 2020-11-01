

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
#include <akari/core/astd.h>
namespace akari::render {
    inline float cos_theta(const glm::vec3 &w) { return w.y; }
    inline float abs_cos_theta(const glm::vec3 &w) { return glm::abs(cos_theta(w)); }
    inline float cos2_theta(const glm::vec3 &w) { return (w.y * w.y); }
    inline float sin2_theta(const glm::vec3 &w) { return (float(1.0) - cos2_theta(w)); }
    inline float sin_theta(const glm::vec3 &w) { return glm::sqrt(glm::max(float(0.0), sin2_theta(w))); }
    inline float tan2_theta(const glm::vec3 &w) { return (sin2_theta(w) / cos2_theta(w)); }
    inline float tan_theta(const glm::vec3 &w) { return glm::sqrt(glm::max(float(0.0), tan2_theta(w))); }
    inline float cos_phi(const glm::vec3 &w) {
        float sinTheta = sin_theta(w);
        return (sinTheta == float(0.0)) ? float(1.0) : glm::clamp((w.x / sinTheta), -float(1.0), float(1.0));
    }
    inline float sin_phi(const glm::vec3 &w) {
        float sinTheta = sin_theta(w);
        return (sinTheta == float(0.0)) ? float(0.0) : glm::clamp((w.z / sinTheta), -float(1.0), float(1.0));
    }
    inline float cos2_phi(const glm::vec3 &w) { return (cos_phi(w) * cos_phi(w)); }
    inline float sin2_phi(const glm::vec3 &w) { return (sin_phi(w) * sin_phi(w)); }
    inline bool same_hemisphere(const glm::vec3 &wo, const glm::vec3 &wi) { return ((wo.y * wi.y) >= float(0.0)); }
    inline std::optional<glm::vec3> refract(const glm::vec3 &wi, const glm::vec3 &n, float eta) {
        float cosThetaI = glm::dot(n, wi);
        float sin2ThetaI = glm::max(float(0.0), (float(1.0) - (cosThetaI * cosThetaI)));
        float sin2ThetaT = ((eta * eta) * sin2ThetaI);
        if ((sin2ThetaT >= float(1.0)))
            return std::nullopt;
        float cosThetaT = glm::sqrt((float(1.0) - sin2ThetaT));
        auto wt = ((eta * -wi) + (((eta * cosThetaI) - cosThetaT) * n));
        return wt;
    }
    inline vec3 faceforward(const vec3 &w, const vec3 &n) { return dot(w, n) < 0.0 ? -n : n; }
    inline float fr_dielectric(float cosThetaI, float etaI, float etaT) {
        bool entering = (cosThetaI > float(0.0));
        if (!entering) {
            std::swap(etaI, etaT);
            cosThetaI = glm::abs(cosThetaI);
        }
        float sinThetaI = glm::sqrt(glm::max(float(0.0), (float(1.0) - (cosThetaI * cosThetaI))));
        float sinThetaT = ((etaI / etaT) * sinThetaI);
        if ((sinThetaT >= float(1.0)))
            return float(1.0);
        float cosThetaT = glm::sqrt(glm::max(float(0.0), (float(1.0) - (sinThetaT * sinThetaT))));
        float Rpar = (((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT)));
        float Rper = (((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT)));
        return (float(0.5) * ((Rpar * Rpar) + (Rper * Rper)));
    }
    inline glm::vec3 fr_conductor(float cosThetaI, const glm::vec3 &etaI, const glm::vec3 &etaT, const glm::vec3 &k) {
        float CosTheta2 = (cosThetaI * cosThetaI);
        float SinTheta2 = (float(1.0) - CosTheta2);
        glm::vec3 Eta = (etaT / etaI);
        glm::vec3 Etak = (k / etaI);
        glm::vec3 Eta2 = (Eta * Eta);
        glm::vec3 Etak2 = (Etak * Etak);
        glm::vec3 t0 = ((Eta2 - Etak2) - SinTheta2);
        glm::vec3 a2plusb2 = glm::sqrt(((t0 * t0) + ((float(4.0) * Eta2) * Etak2)));
        glm::vec3 t1 = (a2plusb2 + CosTheta2);
        glm::vec3 a = glm::sqrt((float(0.5) * (a2plusb2 + t0)));
        glm::vec3 t2 = ((float(2.0) * a) * cosThetaI);
        glm::vec3 Rs = ((t1 - t2) / (t1 + t2));
        glm::vec3 t3 = ((CosTheta2 * a2plusb2) + (SinTheta2 * SinTheta2));
        glm::vec3 t4 = (t2 * SinTheta2);
        glm::vec3 Rp = ((Rs * (t3 - t4)) / (t3 + t4));
        return (float(0.5) * (Rp + Rs));
    }

    inline vec3 spherical_to_xyz(float sinTheta, float cosTheta, float phi) {
        return glm::vec3(sinTheta * glm::cos(phi), cosTheta, sinTheta * glm::sin(phi));
    }

    inline float spherical_theta(const vec3 &v) { return glm::acos(glm::clamp(v.y, -1.0f, 1.0f)); }

    inline float spherical_phi(const glm::vec3 v) {
        float p = glm::atan(v.z, v.x);
        return p < 0.0 ? (p + 2.0 * Pi) : p;
    }
} // namespace akari::render