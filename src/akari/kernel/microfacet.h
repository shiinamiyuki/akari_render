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
#include <akari/kernel/bsdf-funcs.h>
namespace akari {
    AKR_VARIANT
    struct microfacet {
        AKR_IMPORT_TYPES()
        enum MicrofacetType {
            EGGX,
            EBeckmann,
            EPhong,
        };

        AKR_XPU static inline Float BeckmannD(Float alpha, const Vector3f &m) {
            if (m.y() <= 0.0f)
                return 0.0f;
            auto c = bsdf<C>::cos2_theta(m);
            auto t = bsdf<C>::tan2_theta(m);
            auto a2 = alpha * alpha;
            return std::exp(-t / a2) / (Constants<Float>::Pi() * a2 * c * c);
        }

        AKR_XPU static inline Float BeckmannG1(Float alpha, const Vector3f &v, const Normal3f &m) {
            if (dot(v, m) * v.y() <= 0) {
                return 0.0f;
            }
            auto a = 1.0f / (alpha * bsdf<C>::tan_theta(v));
            if (a < 1.6) {
                return (3.535 * a + 2.181 * a * a) / (1.0f + 2.276 * a + 2.577 * a * a);
            } else {
                return 1.0f;
            }
        }
        AKR_XPU static inline Float PhongG1(Float alpha, const Vector3f &v, const Normal3f &m) {
            if (dot(v, m) * v.y() <= 0) {
                return 0.0f;
            }
            auto a = sqrt(0.5f * alpha + 1.0f) / bsdf<C>::tan_theta(v);
            if (a < 1.6) {
                return (3.535 * a + 2.181 * a * a) / (1.0f + 2.276 * a + 2.577 * a * a);
            } else {
                return 1.0f;
            }
        }

        AKR_XPU static inline Float PhongD(Float alpha, const Normal3f &m) {
            if (m.y() <= 0.0f)
                return 0.0f;
            return (alpha + 2.0) / (2.0 * Constants<Float>::Pi()) * pow(m.y(), alpha);
        }

        AKR_XPU static inline Float GGX_D(Float alpha, const Normal3f &m) {
            if (m.y() <= 0.0f)
                return 0.0f;
            Float a2 = alpha * alpha;
            auto c2 = bsdf<C>::cos2_theta(m);
            auto t2 = bsdf<C>::tan2_theta(m);
            auto at = (a2 + t2);
            return a2 / (Constants<Float>::Pi() * c2 * c2 * at * at);
        }

        AKR_XPU static inline Float GGX_G1(Float alpha, const Vector3f &v, const Normal3f &m) {
            if (dot(v, m) * v.y() <= 0) {
                return 0.0f;
            }
            return 2.0 / (1.0 + sqrt(1.0 + alpha * alpha * bsdf<C>::tan2_theta(m)));
        }
        // see https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
        struct MicrofacetModel {
            AKR_XPU MicrofacetModel(MicrofacetType type, Float roughness) : type(type) {
                if (type == EPhong) {
                    alpha = 2.0f / (roughness * roughness) - 2.0f;
                } else {
                    alpha = roughness;
                }
            }
            [[nodiscard]] AKR_XPU Float D(const Normal3f &m) const {
                switch (type) {
                case EBeckmann:
                    return BeckmannD(alpha, m);
                case EPhong:
                    return PhongD(alpha, m);
                case EGGX:
                    return GGX_D(alpha, m);
                }

                return 0.0f;
            }
            [[nodiscard]] AKR_XPU Float G1(const Vector3f &v, const Normal3f &m) const {
                switch (type) {
                case EBeckmann:
                    return BeckmannG1(alpha, v, m);
                case EPhong:
                    return PhongG1(alpha, v, m);
                case EGGX:
                    return GGX_G1(alpha, v, m);
                }
                return 0.0f;
            }
            [[nodiscard]] AKR_XPU Float G(const Vector3f &i, const Vector3f &o, const Normal3f &m) const {
                return G1(i, m) * G1(o, m);
            }
            [[nodiscard]] AKR_XPU Normal3f sample_wh(const Vector3f &wo, const Point2f &u) const {
                Float phi = 2 * Constants<Float>::Pi() * u[1];
                Float cosTheta = 0;
                switch (type) {
                case EBeckmann: {
                    Float t2 = -alpha * alpha * std::log(1 - u[0]);
                    cosTheta = 1.0f / sqrt(1 + t2);
                    break;
                }
                case EPhong: {
                    cosTheta = pow((Float64)u[0], 1.0 / ((Float64)alpha + 2.0f));
                    break;
                }
                case EGGX: {
                    Float t2 = alpha * alpha * u[0] / (1 - u[0]);
                    cosTheta = 1.0f / sqrt(1 + t2);
                    break;
                }
                }
                auto sinTheta = sqrt(std::max(0.0f, 1 - cosTheta * cosTheta));
                auto wh = Normal3f(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
                if (!bsdf<C>::same_hemisphere(wo, wh))
                    wh = -wh;
                return wh;
            }
            [[nodiscard]] AKR_XPU Float evaluate_pdf(const Normal3f &wh) const {
                return D(wh) * bsdf<C>::abs_cos_theta(wh);
            }

          private:
            MicrofacetType type;
            Float alpha;
        };
    };

} // namespace akari