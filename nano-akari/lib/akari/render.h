// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <akari/util.h>
#include <akari/image.h>
#include <akari/scenegraph.h>
#include <array>
namespace akari::scene {
    class SceneGraph;
}
namespace akari::render {
    template <class T>
    struct VarianceTracker {
        std::optional<T> mean, m2;
        int count = 0;
        void update(T value) {
            if (count == 0) {
                mean = value;
                m2 = T(0.0);
            } else {
                auto delta = value - *mean;
                *mean += delta / T(count + 1);
                *m2 += delta * (value - *mean);
            }
            count++;
        }
        std::optional<T> variance() const {
            if (count < 2) {
                return std::nullopt;
            }
            return *m2 / float(count * count);
        }
    };

#pragma region distribution
    /*
     * Return the largest index i such that
     * pred(i) is true
     * If no such index i, last is returned
     * */
    template <typename Pred>
    int upper_bound(int first, int last, Pred pred) {
        int lo = first;
        int hi = last;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (pred(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return std::clamp<int>(hi - 1, 0, (last - first) - 2);
    }

    struct Distribution1D {
        friend struct Distribution2D;
        Distribution1D(const Float *f, size_t n, Allocator<> allocator)
            : func(f, f + n, allocator), cdf(n + 1, allocator) {
            cdf[0] = 0;
            for (size_t i = 0; i < n; i++) {
                cdf[i + 1] = cdf[i] + func[i] / n;
            }
            funcInt = cdf[n];
            if (funcInt == 0) {
                for (uint32_t i = 1; i < n + 1; ++i)
                    cdf[i] = Float(i) / Float(n);
            } else {
                for (uint32_t i = 1; i < n + 1; ++i)
                    cdf[i] /= funcInt;
            }
        }
        // y = F^{-1}(u)
        // P(Y <= y) = P(F^{-1}(U) <= u) = P(U <= F(u)) = F(u)
        // Assume: 0 <= i < n
        [[nodiscard]] Float pdf_discrete(int i) const { return func[i] / (funcInt * count()); }
        [[nodiscard]] Float pdf_continuous(Float x) const {
            uint32_t offset = std::clamp<uint32_t>(static_cast<uint32_t>(x * count()), 0, count() - 1);
            return func[offset] / funcInt;
        }
        std::pair<uint32_t, Float> sample_discrete(Float u) const {
            uint32_t i = upper_bound(0, cdf.size(), [=](int idx) { return cdf[idx] <= u; });
            return {i, pdf_discrete(i)};
        }

        Float sample_continuous(Float u, Float *pdf = nullptr, int *p_offset = nullptr) const {
            uint32_t offset = upper_bound(0, cdf.size(), [=](int idx) { return cdf[idx] <= u; });
            if (p_offset) {
                *p_offset = offset;
            }
            Float du = u - cdf[offset];
            if ((cdf[offset + 1] - cdf[offset]) > 0)
                du /= (cdf[offset + 1] - cdf[offset]);
            if (pdf)
                *pdf = func[offset] / funcInt;
            return ((float)offset + du) / count();
        }

        [[nodiscard]] size_t count() const { return func.size(); }
        [[nodiscard]] Float integral() const { return funcInt; }

      private:
        astd::pmr::vector<Float> func, cdf;
        Float funcInt;
    };

    struct Distribution2D {
        Allocator<> allocator;
        astd::pmr::vector<Distribution1D> pConditionalV;
        std::shared_ptr<Distribution1D> pMarginal;

      public:
        Distribution2D(const Float *data, size_t nu, size_t nv, Allocator<> allocator_)
            : allocator(allocator_), pConditionalV(allocator) {
            pConditionalV.reserve(nv);
            for (auto v = 0u; v < nv; v++) {
                pConditionalV.emplace_back(&data[v * nu], nu, allocator);
            }
            std::vector<Float> m;
            for (auto v = 0u; v < nv; v++) {
                m.emplace_back(pConditionalV[v].funcInt);
            }
            pMarginal = make_pmr_shared<Distribution1D>(allocator, &m[0], nv, allocator);
        }
        Vec2 sample_continuous(const Vec2 &u, Float *pdf) const {
            int v;
            Float pdfs[2];
            auto d1 = pMarginal->sample_continuous(u[0], &pdfs[0], &v);
            auto d0 = pConditionalV[v].sample_continuous(u[1], &pdfs[1]);
            *pdf = pdfs[0] * pdfs[1];
            return Vec2(d0, d1);
        }
        Float pdf_continuous(const Vec2 &p) const {
            auto iu = std::clamp<int>(p[0] * pConditionalV[0].count(), 0, pConditionalV[0].count() - 1);
            auto iv = std::clamp<int>(p[1] * pMarginal->count(), 0, pMarginal->count() - 1);
            return pConditionalV[iv].func[iu] / pMarginal->funcInt;
        }
    };
#pragma endregion
#pragma region sampling
    AKR_XPU inline glm::vec2 concentric_disk_sampling(const glm::vec2 &u) {
        glm::vec2 uOffset = ((float(2.0) * u) - glm::vec2(int32_t(1), int32_t(1)));
        if (((uOffset.x == float(0.0)) && (uOffset.y == float(0.0))))
            return glm::vec2(int32_t(0), int32_t(0));
        float theta = float();
        float r = float();
        if ((glm::abs(uOffset.x) > glm::abs(uOffset.y))) {
            r = uOffset.x;
            theta = (PiOver4 * (uOffset.y / uOffset.x));
        } else {
            r = uOffset.y;
            theta = (PiOver2 - (PiOver4 * (uOffset.x / uOffset.y)));
        }
        return (r * glm::vec2(glm::cos(theta), glm::sin(theta)));
    }
    AKR_XPU inline glm::vec3 cosine_hemisphere_sampling(const glm::vec2 &u) {
        glm::vec2 uv = concentric_disk_sampling(u);
        float r = glm::dot(uv, uv);
        float h = glm::sqrt(glm::max(float(float(0.0)), float((float(1.0) - r))));
        return glm::vec3(uv.x, h, uv.y);
    }
    AKR_XPU inline float cosine_hemisphere_pdf(float cosTheta) { return (cosTheta * InvPi); }
    AKR_XPU inline float uniform_sphere_pdf() { return (float(1.0) / (float(4.0) * Pi)); }
    AKR_XPU inline glm::vec3 uniform_sphere_sampling(const glm::vec2 &u) {
        float z = (float(1.0) - (float(2.0) * u[int32_t(0)]));
        float r = glm::sqrt(glm::max(float(0.0), (float(1.0) - (z * z))));
        float phi = ((float(2.0) * Pi) * u[int32_t(1)]);
        return glm::vec3((r * glm::cos(phi)), (r * glm::sin(phi)), z);
    }
    AKR_XPU inline glm::vec2 uniform_sample_triangle(const glm::vec2 &u) {
        float su0 = glm::sqrt(u[int32_t(0)]);
        float b0 = (float(1.0) - su0);
        float b1 = (u[int32_t(1)] * su0);
        return glm::vec2(b0, b1);
    }
#pragma endregion
#pragma region geometry
    AKR_XPU inline float cos_theta(const glm::vec3 &w) { return w.y; }
    AKR_XPU inline float abs_cos_theta(const glm::vec3 &w) { return glm::abs(cos_theta(w)); }
    AKR_XPU inline float cos2_theta(const glm::vec3 &w) { return (w.y * w.y); }
    AKR_XPU inline float sin2_theta(const glm::vec3 &w) { return (float(1.0) - cos2_theta(w)); }
    AKR_XPU inline float sin_theta(const glm::vec3 &w) { return glm::sqrt(glm::max(float(0.0), sin2_theta(w))); }
    AKR_XPU inline float tan2_theta(const glm::vec3 &w) { return (sin2_theta(w) / cos2_theta(w)); }
    AKR_XPU inline float tan_theta(const glm::vec3 &w) { return glm::sqrt(glm::max(float(0.0), tan2_theta(w))); }
    AKR_XPU inline float cos_phi(const glm::vec3 &w) {
        float sinTheta = sin_theta(w);
        return (sinTheta == float(0.0)) ? float(1.0) : glm::clamp((w.x / sinTheta), -float(1.0), float(1.0));
    }
    AKR_XPU inline float sin_phi(const glm::vec3 &w) {
        float sinTheta = sin_theta(w);
        return (sinTheta == float(0.0)) ? float(0.0) : glm::clamp((w.z / sinTheta), -float(1.0), float(1.0));
    }
    AKR_XPU inline float cos2_phi(const glm::vec3 &w) { return (cos_phi(w) * cos_phi(w)); }
    AKR_XPU inline float sin2_phi(const glm::vec3 &w) { return (sin_phi(w) * sin_phi(w)); }
    AKR_XPU inline bool same_hemisphere(const glm::vec3 &wo, const glm::vec3 &wi) {
        return ((wo.y * wi.y) >= float(0.0));
    }
    AKR_XPU inline std::optional<glm::vec3> refract(const glm::vec3 &wi, const glm::vec3 &n, float eta) {
        float cosThetaI = glm::dot(n, wi);
        float sin2ThetaI = glm::max(float(0.0), (float(1.0) - (cosThetaI * cosThetaI)));
        float sin2ThetaT = ((eta * eta) * sin2ThetaI);
        if ((sin2ThetaT >= float(1.0)))
            return std::nullopt;
        float cosThetaT = glm::sqrt((float(1.0) - sin2ThetaT));
        auto wt = ((eta * -wi) + (((eta * cosThetaI) - cosThetaT) * n));
        return wt;
    }
    AKR_XPU inline vec3 faceforward(const vec3 &w, const vec3 &n) { return dot(w, n) < 0.0 ? -n : n; }
    AKR_XPU inline float fr_dielectric(float cosThetaI, float etaI, float etaT) {
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
    AKR_XPU inline glm::vec3 fr_conductor(float cosThetaI, const glm::vec3 &etaI, const glm::vec3 &etaT,
                                          const glm::vec3 &k) {
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

    AKR_XPU inline vec3 spherical_to_xyz(float sinTheta, float cosTheta, float phi) {
        return glm::vec3(sinTheta * glm::cos(phi), cosTheta, sinTheta * glm::sin(phi));
    }

    AKR_XPU inline float spherical_theta(const vec3 &v) { return glm::acos(glm::clamp(v.y, -1.0f, 1.0f)); }

    AKR_XPU inline float spherical_phi(const glm::vec3 v) {
        float p = glm::atan(v.z, v.x);
        return p < 0.0 ? (p + 2.0 * Pi) : p;
    }
#pragma endregion
#pragma region

    static const int32_t MicrofacetGGX = int32_t(0);
    static const int32_t MicrofacetBeckmann = int32_t(1);
    static const int32_t MicrofacetPhong = int32_t(2);
    AKR_XPU inline float BeckmannD(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        float c = cos2_theta(m);
        float t = tan2_theta(m);
        float a2 = (alpha * alpha);
        return (glm::exp((-t / a2)) / (((Pi * a2) * c) * c));
    }
    AKR_XPU inline float BeckmannG1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        float a = (float(1.0) / (alpha * tan_theta(v)));
        if ((a < float(1.6))) {
            return (((float(3.535) * a) + ((float(2.181) * a) * a)) /
                    ((float(1.0) + (float(2.276) * a)) + ((float(2.577) * a) * a)));
        } else {
            return float(1.0);
        }
    }
    AKR_XPU inline float PhongG1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        float a = (glm::sqrt(((float(0.5) * alpha) + float(1.0))) / tan_theta(v));
        if ((a < float(1.6))) {
            return (((float(3.535) * a) + ((float(2.181) * a) * a)) /
                    ((float(1.0) + (float(2.276) * a)) + ((float(2.577) * a) * a)));
        } else {
            return float(1.0);
        }
    }
    AKR_XPU inline float PhongD(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        return (((alpha + float(2.0)) / (float(2.0) * Pi)) * glm::pow(m.y, alpha));
    }
    AKR_XPU inline float GGX_D(float alpha, const glm::vec3 &m) {
        if ((m.y <= float(0.0)))
            return float(0.0);
        float a2 = (alpha * alpha);
        float c2 = cos2_theta(m);
        float t2 = tan2_theta(m);
        float at = (a2 + t2);
        return (a2 / ((((Pi * c2) * c2) * at) * at));
    }
    AKR_XPU inline float GGX_G1(float alpha, const glm::vec3 &v, const glm::vec3 &m) {
        if (((glm::dot(v, m) * v.y) <= float(0.0))) {
            return float(0.0);
        }
        return (float(2.0) / (float(1.0) + glm::sqrt((float(1.0) + ((alpha * alpha) * tan2_theta(m))))));
    }
    struct MicrofacetModel {
        int32_t type;
        float alpha;
    };
    AKR_XPU inline MicrofacetModel microfacet_new(int32_t type, float roughness) {
        float alpha = float();
        if ((type == MicrofacetPhong)) {
            alpha = ((float(2.0) / (roughness * roughness)) - float(2.0));
        } else {
            alpha = roughness;
        }
        return MicrofacetModel{type, alpha};
    }
    AKR_XPU inline float microfacet_D(const MicrofacetModel &model, const glm::vec3 &m) {
        int32_t type = model.type;
        float alpha = model.alpha;
        switch (type) {
        case MicrofacetBeckmann: {
            return BeckmannD(alpha, m);
        }
        case MicrofacetPhong: {
            return PhongD(alpha, m);
        }
        case MicrofacetGGX: {
            return GGX_D(alpha, m);
        }
        }
        return float(0.0);
    }
    AKR_XPU inline float microfacet_G1(const MicrofacetModel &model, const glm::vec3 &v, const glm::vec3 &m) {
        int32_t type = model.type;
        float alpha = model.alpha;
        switch (type) {
        case MicrofacetBeckmann: {
            return BeckmannG1(alpha, v, m);
        }
        case MicrofacetPhong: {
            return PhongG1(alpha, v, m);
        }
        case MicrofacetGGX: {
            return GGX_G1(alpha, v, m);
        }
        }
        return float(0.0);
    }
    AKR_XPU inline float microfacet_G(const MicrofacetModel &model, const glm::vec3 &i, const glm::vec3 &o,
                                      const glm::vec3 &m) {
        return (microfacet_G1(model, i, m) * microfacet_G1(model, o, m));
    }
    AKR_XPU inline glm::vec3 microfacet_sample_wh(const MicrofacetModel &model, const glm::vec3 &wo,
                                                  const glm::vec2 &u) {
        int32_t type = model.type;
        float alpha = model.alpha;
        float phi = ((float(2.0) * Pi) * u.y);
        float cosTheta = float(0.0);
        switch (type) {
        case MicrofacetBeckmann: {
            float t2 = ((-alpha * alpha) * glm::log((float(1.0) - u.x)));
            cosTheta = (float(1.0) / glm::sqrt((float(1.0) + t2)));
            break;
        }
        case MicrofacetPhong: {
            cosTheta = glm::pow(u.x, float((float(1.0) / (alpha + float(2.0)))));
            break;
        }
        case MicrofacetGGX: {
            float t2 = (((alpha * alpha) * u.x) / (float(1.0) - u.x));
            cosTheta = (float(1.0) / glm::sqrt((float(1.0) + t2)));
            break;
        }
        }
        float sinTheta = glm::sqrt(glm::max(float(0.0), (float(1.0) - (cosTheta * cosTheta))));
        glm::vec3 wh = glm::vec3((glm::cos(phi) * sinTheta), cosTheta, (glm::sin(phi) * sinTheta));
        if (!same_hemisphere(wo, wh))
            wh = -wh;
        return wh;
    }
    AKR_XPU inline float microfacet_evaluate_pdf(const MicrofacetModel &m, const glm::vec3 &wh) {
        return (microfacet_D(m, wh) * abs_cos_theta(wh));
    }
#pragma endregion
    struct Rng {
        Rng(uint64_t sequence = 0) { pcg32_init(sequence); }
        uint32_t uniform_u32() { return pcg32(); }
        double uniform_float() { return pcg32() / double(0xffffffff); }

      private:
        uint64_t state = 0x4d595df4d0f33173; // Or something seed-dependent
        static uint64_t const multiplier = 6364136223846793005u;
        static uint64_t const increment = 1442695040888963407u; // Or an arbitrary odd constant
        static uint32_t rotr32(uint32_t x, unsigned r) { return x >> r | x << (-r & 31); }
        uint32_t pcg32(void) {
            uint64_t x = state;
            unsigned count = (unsigned)(x >> 59); // 59 = 64 - 5

            state = x * multiplier + increment;
            x ^= x >> 18;                              // 18 = (64 - 27)/2
            return rotr32((uint32_t)(x >> 27), count); // 27 = 32 - 5
        }
        void pcg32_init(uint64_t seed) {
            state = seed + increment;
            (void)pcg32();
        }
    };
    class PCGSampler {
        Rng rng;

      public:
        void set_sample_index(uint64_t idx) { rng = Rng(idx); }
        Float next1d() { return rng.uniform_float(); }

        void start_next_sample() {}
        PCGSampler(uint64_t seed = 0u) : rng(seed) {}
    };
    class LCGSampler {
        uint32_t seed;

      public:
        void set_sample_index(uint64_t idx) { seed = idx & 0xffffffff; }
        Float next1d() {
            seed = (1103515245 * seed + 12345);
            return (Float)seed / (Float)0xFFFFFFFF;
        }
        void start_next_sample() {}
        LCGSampler(uint64_t seed = 0u) : seed(seed) {}
    };

    struct MLTSampler {
        struct PrimarySample {
            Float value;
            Float _backup;
            uint64_t last_modification_iteration;
            uint64_t last_modified_backup;

            void backup() {
                _backup = value;
                last_modified_backup = last_modification_iteration;
            }

            void restore() {
                value = _backup;
                last_modification_iteration = last_modified_backup;
            }
        };
        explicit MLTSampler(unsigned int seed) : rng(seed) {}
        Rng rng;
        std::vector<PrimarySample> X;
        uint64_t current_iteration = 0;
        bool large_step = true;
        uint64_t last_large_step = 0;
        Float large_step_prob = 0.25;
        uint32_t sample_index = 0;
        uint64_t accepts = 0, rejects = 0;
        Float uniform() { return rng.uniform_float(); }
        void start_next_sample() {
            sample_index = 0;
            current_iteration++;
            large_step = uniform() < large_step_prob;
        }
        void set_sample_index(uint64_t idx) { AKR_PANIC("shouldn't be called"); }
        Float next1d() {
            if (sample_index >= X.size()) {
                X.resize(sample_index + 1u);
            }
            auto &Xi = X[sample_index];
            mutate(Xi, sample_index);
            sample_index += 1;
            return Xi.value;
        }
        double mutate(double x, double s1, double s2) {
            double r = uniform();
            if (r < 0.5) {
                r = r * 2.0;
                x = x + s2 * exp(-log(s2 / s1) * r);
                if (x > 1.0)
                    x -= 1.0;
            } else {
                r = (r - 0.5) * 2.0;
                x = x - s2 * exp(-log(s2 / s1) * r);
                if (x < 0.0)
                    x += 1.0;
            }
            return x;
        }
        void mutate(PrimarySample &Xi, int sampleIndex) {
            double s1, s2;
            s1 = 1.0 / 1024.0, s2 = 1.0 / 64.0;

            if (Xi.last_modification_iteration < last_large_step) {
                Xi.value = uniform();
                Xi.last_modification_iteration = last_large_step;
            }

            if (large_step) {
                Xi.backup();
                Xi.value = uniform();
            } else {
                int64_t nSmall = current_iteration - Xi.last_modification_iteration;

                auto nSmallMinus = nSmall - 1;
                if (nSmallMinus > 0) {
                    auto x = Xi.value;
                    while (nSmallMinus > 0) {
                        nSmallMinus--;
                        x = mutate(x, s1, s2);
                    }
                    Xi.value = x;
                    Xi.last_modification_iteration = current_iteration - 1;
                }
                Xi.backup();
                Xi.value = mutate(Xi.value, s1, s2);
            }

            Xi.last_modification_iteration = current_iteration;
        }
        void accept() {
            if (large_step) {
                last_large_step = current_iteration;
            }
            accepts++;
        }

        void reject() {
            for (PrimarySample &Xi : X) {
                if (Xi.last_modification_iteration == current_iteration) {
                    Xi.restore();
                }
            }
            rejects++;
            --current_iteration;
        }
    };

    struct ReplaySampler {
        explicit ReplaySampler(astd::pmr::vector<Float> Xs, Rng rng) : rng(rng), Xs(std::move(Xs)) {}
        Float next1d() {
            if (idx < Xs.size()) {
                return Xs[idx++];
            }
            idx++;
            return rng.uniform_float();
        }
        void start_next_sample() { idx = 0; }
        void set_sample_index(uint64_t) {}

      private:
        uint32_t idx = 0;
        Rng rng;
        astd::pmr::vector<Float> Xs;
    };
    struct Sampler : Variant<LCGSampler, PCGSampler, MLTSampler, ReplaySampler> {
        using Variant::Variant;
        Sampler() : Sampler(PCGSampler()) {}
        Float next1d() { AKR_VAR_DISPATCH(next1d); }
        vec2 next2d() { return vec2(next1d(), next1d()); }
        void start_next_sample() { AKR_VAR_DISPATCH(start_next_sample); }
        void set_sample_index(uint64_t idx) { AKR_VAR_DISPATCH(set_sample_index, idx); }
    };

    struct Film {
        Array2D<Spectrum> radiance;
        Array2D<std::array<AtomicFloat, Spectrum::size>> splats;
        Array2D<Float> weight;
        explicit Film(const ivec2 &dimension) : radiance(dimension), weight(dimension), splats(dimension) {}
        void add_sample(const ivec2 &p, const Spectrum &sample, Float weight_) {
            weight(p) += weight_;
            radiance(p) += sample;
        }
        void splat(const ivec2 &p, const Spectrum &sample) {
            for (size_t i = 0; i < Spectrum::size; i++) {
                splats(p)[i].add(sample[i]);
            }
        }
        [[nodiscard]] ivec2 resolution() const { return radiance.dimension(); }
        Array2D<Spectrum> to_array2d() const {
            Array2D<Spectrum> array(resolution());
            thread::parallel_for(resolution().y, [&](uint32_t y, uint32_t) {
                for (int x = 0; x < resolution().x; x++) {
                    Spectrum splat_s;
                    for (size_t i = 0; i < Spectrum::size; i++) {
                        splat_s[i] = splats(x, y)[i].value();
                    }
                    if (weight(x, y) != 0) {
                        auto color = (radiance(x, y)) / weight(x, y);
                        array(x, y) = color + splat_s;
                    } else {
                        auto color = radiance(x, y);
                        array(x, y) = color + splat_s;
                    }
                }
            });
            return array;
        }
        template <typename = std::enable_if_t<std::is_same_v<Spectrum, Color3f>>>
        Image to_rgb_image() const {
            Image image = rgb_image(resolution());
            thread::parallel_for(resolution().y, [&](uint32_t y, uint32_t) {
                for (int x = 0; x < resolution().x; x++) {
                    Spectrum splat_s;
                    for (size_t i = 0; i < Spectrum::size; i++) {
                        splat_s[i] = splats(x, y)[i].value();
                    }
                    if (weight(x, y) != 0) {
                        auto color = (radiance(x, y)) / weight(x, y) + splat_s;
                        image(x, y, 0) = color[0];
                        image(x, y, 1) = color[1];
                        image(x, y, 2) = color[2];
                    } else {
                        auto color = radiance(x, y) + splat_s;
                        image(x, y, 0) = color[0];
                        image(x, y, 1) = color[1];
                        image(x, y, 2) = color[2];
                    }
                }
            });
            return image;
        }
    };
    struct CameraSample {
        vec2 p_lens;
        vec2 p_film;
        Float weight = 0.0f;
        Vec3 normal;
        Ray ray;
    };
    struct PerspectiveCamera {
        Transform c2w, w2c, r2c, c2r;
        ivec2 _resolution;
        Float fov;
        Float lens_radius = 0.0f;
        Float focal_distance = 0.0f;
        PerspectiveCamera(const ivec2 &_resolution, const Transform &c2w, Float fov)
            : c2w(c2w), w2c(c2w.inverse()), _resolution(_resolution), fov(fov) {
            preprocess();
        }
        ivec2 resolution() const { return _resolution; }
        CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
            CameraSample sample;
            sample.p_lens = concentric_disk_sampling(u1) * lens_radius;
            sample.p_film = vec2(raster) + u2;
            sample.weight = 1;

            vec2 p = shuffle<0, 1>(r2c.apply_point(Vec3(sample.p_film.x, sample.p_film.y, 0.0f)));
            Ray ray(Vec3(0), Vec3(normalize(Vec3(p.x, p.y, 0) - Vec3(0, 0, 1))));
            if (lens_radius > 0 && focal_distance > 0) {
                Float ft = focal_distance / std::abs(ray.d.z);
                Vec3 pFocus = ray(ft);
                ray.o = Vec3(sample.p_lens.x, sample.p_lens.y, 0);
                ray.d = Vec3(normalize(pFocus - ray.o));
            }
            ray.o = c2w.apply_point(ray.o);
            ray.d = c2w.apply_vector(ray.d);
            sample.normal = c2w.apply_normal(Vec3(0, 0, -1.0f));
            sample.ray = ray;

            return sample;
        }

      private:
        void preprocess() {
            Transform m;
            m = Transform::scale(Vec3(1.0f / _resolution.x, 1.0f / _resolution.y, 1)) * m;
            m = Transform::scale(Vec3(2, 2, 1)) * m;
            m = Transform::translate(Vec3(-1, -1, 0)) * m;
            m = Transform::scale(Vec3(1, -1, 1)) * m;
            auto s = atan(fov / 2);
            if (_resolution.x > _resolution.y) {
                m = Transform::scale(Vec3(s, s * Float(_resolution.y) / _resolution.x, 1)) * m;
            } else {
                m = Transform::scale(Vec3(s * Float(_resolution.x) / _resolution.y, s, 1)) * m;
            }
            r2c = m;
            c2r = r2c.inverse();
        }
    };
    struct Camera : Variant<PerspectiveCamera> {
        using Variant::Variant;
        ivec2 resolution() const { AKR_VAR_DISPATCH(resolution); }
        CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
            AKR_VAR_DISPATCH(generate_ray, u1, u2, raster);
        }
    };
    struct ShadingPoint {
        Vec2 texcoords;
        Vec3 p;
        Vec3 dpdu, dpdv;
        Vec3 n;
        Vec3 dndu, dndv;
        Vec3 ng;
        ShadingPoint() = default;
        ShadingPoint(Vec2 tc) : texcoords(tc) {}
    };

    struct ConstantTexture {
        Spectrum value;
        ConstantTexture(Float v) : value(v) {}
        ConstantTexture(Spectrum v) : value(v) {}
        Float evaluate_f(const ShadingPoint &sp) const { return value[0]; }
        Spectrum evaluate_s(const ShadingPoint &sp) const { return value; }
    };

    struct DeviceImageImpl;
    using DeviceImage = DeviceImageImpl *;
    struct ImageTexture {
        DeviceImage image;
    };

    struct Texture : Variant<ConstantTexture> {
        using Variant::Variant;
        Float evaluate_f(const ShadingPoint &sp) const { AKR_VAR_DISPATCH(evaluate_f, sp); }
        Spectrum evaluate_s(const ShadingPoint &sp) const { AKR_VAR_DISPATCH(evaluate_s, sp); }
        Texture() : Texture(ConstantTexture(0.0)) {}
    };

    enum class BSDFType : int {
        Unset = 0u,
        Reflection = 1u << 0,
        Transmission = 1u << 1,
        Diffuse = 1u << 2,
        Glossy = 1u << 3,
        Specular = 1u << 4,
        DiffuseReflection = Diffuse | Reflection,
        DiffuseTransmission = Diffuse | Transmission,
        GlossyReflection = Glossy | Reflection,
        GlossyTransmission = Glossy | Transmission,
        SpecularReflection = Specular | Reflection,
        SpecularTransmission = Specular | Transmission,
        All = Diffuse | Glossy | Specular | Reflection | Transmission
    };
    AKR_XPU inline BSDFType operator&(BSDFType a, BSDFType b) { return BSDFType((int)a & (int)b); }
    AKR_XPU inline BSDFType operator|(BSDFType a, BSDFType b) { return BSDFType((int)a | (int)b); }
    AKR_XPU inline BSDFType operator~(BSDFType a) { return BSDFType(~(uint32_t)a); }
    struct BSDFSample {
        Vec3 wi;
        Spectrum f;
        Float pdf = 0.0;
        BSDFType type = BSDFType::Unset;
    };
    class BSDFClosure;
    class DiffuseBSDF {
        Spectrum R;

      public:
        DiffuseBSDF(const Spectrum &R) : R(R) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {

            if (same_hemisphere(wo, wi)) {
                return cosine_hemisphere_pdf(std::abs(cos_theta(wi)));
            }
            return 0.0f;
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const {

            if (same_hemisphere(wo, wi)) {
                return R * InvPi;
            }
            return Spectrum(0.0f);
        }
        [[nodiscard]] BSDFType type() const { return BSDFType::DiffuseReflection; }
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            sample.wi = cosine_hemisphere_sampling(u);
            if (!same_hemisphere(wo, sample.wi)) {
                sample.wi.y = -sample.wi.y;
            }
            sample.type = type();
            sample.pdf = cosine_hemisphere_pdf(std::abs(cos_theta(sample.wi)));
            sample.f = R * InvPi;
            return sample;
        }
    };
    class FresnelNoOp {
      public:
        [[nodiscard]] Spectrum evaluate(Float cosThetaI) const;
    };

    class FresnelConductor {
        const Spectrum etaI, etaT, k;

      public:
        FresnelConductor(const Spectrum &etaI, const Spectrum &etaT, const Spectrum &k)
            : etaI(etaI), etaT(etaT), k(k) {}
        [[nodiscard]] Spectrum evaluate(Float cosThetaI) const;
    };
    class FresnelDielectric {
        Float etaI, etaT;

      public:
        FresnelDielectric(const Float &etaI, const Float &etaT) : etaI(etaI), etaT(etaT) {}
        [[nodiscard]] Spectrum evaluate(Float cosThetaI) const;
    };
    class Fresnel : public Variant<FresnelConductor, FresnelDielectric, FresnelNoOp> {
      public:
        using Variant::Variant;
        Fresnel() : Variant(FresnelNoOp()) {}
        [[nodiscard]] Spectrum evaluate(Float cosThetaI) const { AKR_VAR_DISPATCH(evaluate, cosThetaI); }
    };
    class SpecularReflection {
        Spectrum R;

      public:
        SpecularReflection(const Spectrum &R) : R(R) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.0f; }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const { return Spectrum(0.0f); }
        [[nodiscard]] BSDFType type() const { return BSDFType::SpecularReflection; }
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            sample.wi = glm::reflect(-wo, vec3(0, 1, 0));
            sample.type = type();
            sample.pdf = 1.0;
            sample.f = R / (std::abs(cos_theta(sample.wi)));
            return sample;
        }
    };
    class SpecularTransmission {
        Spectrum R;
        Float eta;

      public:
        SpecularTransmission(const Spectrum &R, Float eta) : R(R), eta(eta) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.0f; }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const { return Spectrum(0.0f); }
        [[nodiscard]] BSDFType type() const { return BSDFType::SpecularTransmission; }
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            Float etaIO = same_hemisphere(wo, vec3(0, 1, 0)) ? eta : 1.0f / eta;
            auto wt = refract(wo, faceforward(wo, vec3(0, 1, 0)), etaIO);
            if (!wt) {
                return std::nullopt;
            }
            sample.wi = *wt;
            sample.type = type();
            sample.pdf = 1.0;
            sample.f = R / (std::abs(cos_theta(sample.wi)));
            return sample;
        }
    };

    class AKR_EXPORT FresnelSpecular {
        Spectrum R, T;
        Float etaA, etaB;
        FresnelDielectric fresnel;

      public:
        explicit FresnelSpecular(const Spectrum &R, const Spectrum &T, Float etaA, Float etaB)
            : R(R), T(T), etaA(etaA), etaB(etaB), fresnel(etaA, etaB) {}
        [[nodiscard]] BSDFType type() const { return BSDFType::SpecularTransmission | BSDFType::SpecularReflection; }
        [[nodiscard]] Float evaluate_pdf(const vec3 &wo, const vec3 &wi) const { return 0; }
        [[nodiscard]] Spectrum evaluate(const vec3 &wo, const vec3 &wi) const { return Spectrum(0); }
        [[nodiscard]] std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const;
    };
    class MixBSDF {
      public:
        Float fraction;
        const BSDFClosure *bsdf_A = nullptr;
        const BSDFClosure *bsdf_B = nullptr;
        MixBSDF(Float fraction, const BSDFClosure *bsdf_A, const BSDFClosure *bsdf_B)
            : fraction(fraction), bsdf_A(bsdf_A), bsdf_B(bsdf_B) {}
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const;
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const;
        [[nodiscard]] BSDFType type() const;
        std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const;
    };
    class BSDFClosure
        : public Variant<DiffuseBSDF, SpecularReflection, SpecularTransmission, FresnelSpecular, MixBSDF> {
      public:
        using Variant::Variant;
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            AKR_VAR_DISPATCH(evaluate_pdf, wo, wi);
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const { AKR_VAR_DISPATCH(evaluate, wo, wi); }
        [[nodiscard]] BSDFType type() const { AKR_VAR_DISPATCH(type); }
        [[nodiscard]] bool match_flags(BSDFType flag) const { return ((uint32_t)type() & (uint32_t)flag) != 0; }
        [[nodiscard]] std::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            AKR_VAR_DISPATCH(sample, u, wo);
        }
    };
    struct BSDFSampleContext {
        Float u0;
        Vec2 u1;
        const Vec3 wo;
    };
    class BSDF {
        std::optional<BSDFClosure> closure_;
        Frame frame;
        Float choice_pdf = 1.0f;

      public:
        BSDF(const Frame &frame) : frame(frame) {}
        bool null() const { return !closure_.has_value(); }
        void set_closure(const BSDFClosure &closure) { closure_ = closure; }
        void set_choice_pdf(Float pdf) { choice_pdf = pdf; }
        const BSDFClosure &closure() const { return *closure_; }
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            auto pdf = closure().evaluate_pdf(frame.world_to_local(wo), frame.world_to_local(wi));
            return pdf * choice_pdf;
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const {
            auto f = closure().evaluate(frame.world_to_local(wo), frame.world_to_local(wi));
            return f;
        }

        [[nodiscard]] BSDFType type() const { return closure().type(); }
        [[nodiscard]] bool match_flags(BSDFType flag) const { return closure().match_flags(flag); }
        std::optional<BSDFSample> sample(const BSDFSampleContext &ctx) const {
            auto wo = frame.world_to_local(ctx.wo);
            if (auto sample = closure().sample(ctx.u1, wo)) {
                sample->wi = frame.local_to_world(sample->wi);
                sample->pdf *= choice_pdf;
                return sample;
            }
            return std::nullopt;
        }
    };

    struct Light;
    struct Material;
    struct Triangle {
        std::array<Vec3, 3> vertices;
        std::array<Vec3, 3> normals;
        std::array<vec2, 3> texcoords;
        const Material *material = nullptr;
        const Light *light = nullptr;
        Vec3 p(const vec2 &uv) const { return lerp3(vertices[0], vertices[1], vertices[2], uv); }
        Float area() const { return length(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) * 0.5f; }
        Vec3 ng() const { return normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])); }
        Vec3 ns(const vec2 &uv) const { return normalize(lerp3(normals[0], normals[1], normals[2], uv)); }
        vec2 texcoord(const vec2 &uv) const { return lerp3(texcoords[0], texcoords[1], texcoords[2], uv); }
        Vec3 dpdu(Float u) const { return dlerp3du(vertices[0], vertices[1], vertices[2], u); }
        Vec3 dpdv(Float v) const { return dlerp3du(vertices[0], vertices[1], vertices[2], v); }

        std::pair<Vec3, Vec3> dnduv(const vec2 &uv) const {
            auto n = ns(uv);
            Float il = 1.0 / length(n);
            n *= il;
            auto dn_du = (normals[1] - normals[0]) * il;
            auto dn_dv = (normals[2] - normals[0]) * il;
            dn_du = -n * dot(n, dn_du) + dn_du;
            dn_dv = -n * dot(n, dn_dv) + dn_dv;
            return std::make_pair(dn_du, dn_dv);
        }

        std::optional<std::pair<Float, Vec2>> intersect(const Ray &ray) const {
            auto &v0 = vertices[0];
            auto &v1 = vertices[1];
            auto &v2 = vertices[2];
            Vec3 e1 = (v1 - v0);
            Vec3 e2 = (v2 - v0);
            Float a, f, u, v;
            auto h = cross(ray.d, e2);
            a = dot(e1, h);
            if (a > Float(-1e-6f) && a < Float(1e-6f))
                return std::nullopt;
            f = 1.0f / a;
            auto s = ray.o - v0;
            u = f * dot(s, h);
            if (u < 0.0 || u > 1.0)
                return std::nullopt;
            auto q = cross(s, e1);
            v = f * dot(ray.d, q);
            if (v < 0.0 || u + v > 1.0)
                return std::nullopt;
            Float t = f * dot(e2, q);
            if (t > ray.tmin && t < ray.tmax) {
                return std::make_pair(t, Vec2(u, v));
            } else {
                return std::nullopt;
            }
        }
    };
    struct Material;
    struct MeshInstance;
    struct SurfaceInteraction {
        Triangle triangle;
        Vec3 p;
        Vec3 ng, ns;
        vec2 texcoords;
        Vec3 dndu, dndv;
        Vec3 dpdu, dpdv;
        const MeshInstance *shape = nullptr;
        SurfaceInteraction(const vec2 &uv, const Triangle &triangle)
            : triangle(triangle), p(triangle.p(uv)), ng(triangle.ng()), ns(triangle.ns(uv)),
              texcoords(triangle.texcoord(uv)) {
            dpdu = triangle.dpdu(uv[0]);
            dpdv = triangle.dpdu(uv[1]);
            std::tie(dndu, dndv) = triangle.dnduv(uv);
        }
        const Light *light() const { return triangle.light; }
        const Material *material() const { return triangle.material; }
        ShadingPoint sp() const {
            ShadingPoint sp_;
            sp_.n = ns;
            sp_.texcoords = texcoords;
            sp_.dndu = dndu;
            sp_.dndv = dndv;
            sp_.dpdu = dpdu;
            sp_.dpdv = dpdv;
            return sp_;
        }
    };

    struct Material {
        Texture color;
        Texture metallic;
        Texture roughness;
        Texture specular;
        Texture emission;
        Texture transmission;
        Material() {}
        BSDF evaluate(Sampler &sampler, Allocator<> alloc, const SurfaceInteraction &si) const;
    };

    struct MeshInstance {
        Transform transform;
        BufferView<const vec3> vertices;
        BufferView<const ivec3> indices;
        BufferView<const vec3> normals;
        BufferView<const vec2> texcoords;
        std::vector<const Light *> lights;
        const scene::Mesh *mesh = nullptr;
        const Material *material = nullptr;

        Triangle get_triangle(int prim_id) const {
            Triangle trig;
            for (int i = 0; i < 3; i++) {
                trig.vertices[i] = transform.apply_vector(vertices[indices[prim_id][i]]);
                trig.normals[i] = transform.apply_normal(normals[indices[prim_id][i]]);
                if (!texcoords.empty())
                    trig.texcoords[i] = texcoords[indices[prim_id][i]];
                else {
                    trig.texcoords[i] = vec2(i > 1, i % 2 == 0);
                }
            }
            trig.material = material;
            if (!lights.empty()) {
                trig.light = lights[prim_id];
            }
            return trig;
        }
    };
    struct PointGeometry {
        Vec3 p;
        Vec3 n;
    };

    struct LightSampleContext {
        vec2 u;
        Vec3 p;
    };
    struct LightSample {
        Vec3 ng = Vec3(0.0f);
        Vec3 wi; // w.r.t to the luminated surface; normalized
        Float pdf = 0.0f;
        Spectrum I;
        Ray shadow_ray;
    };
    struct LightRaySample {
        Ray ray;
        Spectrum E;
        vec3 ng;
        vec2 uv; // 2D parameterized position
        Float pdfPos = 0.0, pdfDir = 0.0;
    };

    struct AreaLight {
        Triangle triangle;
        Texture color;
        bool double_sided = false;
        AreaLight(Triangle triangle, Texture color, bool double_sided)
            : triangle(triangle), color(color), double_sided(double_sided) {
            ng = triangle.ng();
        }
        Spectrum Le(const Vec3 &wo, const ShadingPoint &sp) const {
            bool face_front = dot(wo, ng) > 0.0;
            if (double_sided || face_front) {
                return color.evaluate_s(sp);
            }
            return Spectrum(0.0);
        }
        Float pdf_incidence(const PointGeometry &ref, const vec3 &wi) const {
            Ray ray(ref.p, wi);
            auto hit = triangle.intersect(ray);
            if (!hit) {
                return 0.0f;
            }
            Float SA = triangle.area() * (-glm::dot(wi, triangle.ng())) / (hit->first * hit->first);
            return 1.0f / SA;
        }
        LightSample sample_incidence(const LightSampleContext &ctx) const {
            auto coords = uniform_sample_triangle(ctx.u);
            auto p = triangle.p(coords);
            LightSample sample;
            sample.ng = triangle.ng();
            sample.wi = p - ctx.p;
            auto dist_sqr = dot(sample.wi, sample.wi);
            sample.wi /= sqrt(dist_sqr);
            sample.I = color.evaluate_s(ShadingPoint(triangle.texcoord(coords)));
            sample.pdf = dist_sqr / max(Float(0.0), -dot(sample.wi, sample.ng)) / triangle.area();
            sample.shadow_ray = Ray(p, -sample.wi, Eps / std::abs(dot(sample.wi, sample.ng)),
                                    sqrt(dist_sqr) * (Float(1.0f) - ShadowEps));
            return sample;
        }

      private:
        Vec3 ng;
    };
    struct Light : Variant<AreaLight> {
        using Variant::Variant;
        Spectrum Le(const Vec3 &wo, const ShadingPoint &sp) const { AKR_VAR_DISPATCH(Le, wo, sp); }
        Float pdf_incidence(const PointGeometry &ref, const vec3 &wi) const {
            AKR_VAR_DISPATCH(pdf_incidence, ref, wi);
        }
        LightSample sample_incidence(const LightSampleContext &ctx) const { AKR_VAR_DISPATCH(sample_incidence, ctx); }
    };
    struct Intersection {
        Float t = Inf;
        Vec2 uv;
        int geom_id = -1;
        int prim_id = -1;
        bool hit() const { return geom_id != -1; }
    };
    struct Scene;
    class EmbreeAccel {
      public:
        virtual void build(const Scene &scene, const std::shared_ptr<scene::SceneGraph> &scene_graph) = 0;
        virtual std::optional<Intersection> intersect1(const Ray &ray) const = 0;
        virtual bool occlude1(const Ray &ray) const = 0;
        virtual Bounds3f world_bounds() const = 0;
    };
    std::shared_ptr<EmbreeAccel> create_embree_accel();

    struct PowerLightSampler {
        PowerLightSampler(Allocator<> alloc, BufferView<const Light *> lights_, const std::vector<Float> &power)
            : light_distribution(power.data(), power.size(), alloc), lights(lights_) {
            for (uint32_t i = 0; i < power.size(); i++) {
                light_pdf[lights[i]] = light_distribution.pdf_discrete(i);
            }
        }
        Distribution1D light_distribution;
        BufferView<const Light *> lights;
        std::unordered_map<const Light *, Float> light_pdf;
        std::pair<const Light *, Float> sample(Vec2 u) const {
            auto [light_idx, pdf] = light_distribution.sample_discrete(u[0]);
            return std::make_pair(lights[light_idx], pdf);
        }
        Float pdf(const Light *light) const {
            auto it = light_pdf.find(light);
            if (it == light_pdf.end()) {
                return 0.0;
            }
            return it->second;
        }
    };
    struct LightSampler : Variant<std::shared_ptr<PowerLightSampler>> {
        using Variant::Variant;
        std::pair<const Light *, Float> sample(Vec2 u) const { AKR_VAR_PTR_DISPATCH(sample, u); }
        Float pdf(const Light *light) const { AKR_VAR_PTR_DISPATCH(pdf, light); }
    };
    struct Scene {
        std::optional<Camera> camera;
        std::vector<MeshInstance> instances;
        std::vector<const Material *> materials;
        std::vector<const Light *> lights;
        std::shared_ptr<EmbreeAccel> accel;
        Allocator<> allocator;
        std::optional<LightSampler> light_sampler;
        astd::pmr::monotonic_buffer_resource *rsrc;
        std::optional<SurfaceInteraction> intersect(const Ray &ray) const;
        bool occlude(const Ray &ray) const;
        Scene() = default;
        Scene(const Scene &) = delete;
        ~Scene();
    };

    std::shared_ptr<const Scene> create_scene(Allocator<>, const std::shared_ptr<scene::SceneGraph> &scene_graph);

    struct PTConfig {
        Sampler sampler;
        int min_depth = 3;
        int max_depth = 5;
        int spp = 16;
    };
    Film render_pt(PTConfig config, const Scene &scene);

    // separate emitter direct hit
    // useful for MLT
    std::pair<Spectrum, Spectrum> render_pt_pixel_separete_emitter_direct(PTConfig config, Allocator<>,
                                                                          const Scene &scene, Sampler &sampler,
                                                                          const vec2 &p_film);
    inline Spectrum render_pt_pixel_wo_emitter_direct(PTConfig config, Allocator<> allocator, const Scene &scene,
                                                      Sampler &sampler, const vec2 &p_film) {
        auto [_, rest] = render_pt_pixel_separete_emitter_direct(config, allocator, scene, sampler, p_film);
        return rest - _;
    }
    inline Spectrum render_pt_pixel(PTConfig config, Allocator<> allocator, const Scene &scene, Sampler &sampler,
                                    const vec2 &p_film) {
        auto [_, rest] = render_pt_pixel_separete_emitter_direct(config, allocator, scene, sampler, p_film);
        return rest;
    }
    struct SMSConfig {
        Sampler sampler;
        int min_depth = 3;
        int max_depth = 5;
        int spp = 16;
    };

    // sms single scatter
    Film render_sms_ss(SMSConfig config, const Scene &scene);
    struct BDPTConfig {
        Sampler sampler;
        int min_depth = 3;
        int max_depth = 5;
        int spp = 16;
    };

    Film render_bdpt(PTConfig config, const Scene &scene);

    struct MLTConfig {
        int num_bootstrap = 100000;
        int num_chains = 1024;
        int min_depth = 3;
        int max_depth = 5;
        int spp = 16;
    };
    Image render_mlt(MLTConfig config, const Scene &scene);
    Image render_smcmc(MLTConfig config, const Scene &scene);
} // namespace akari::render