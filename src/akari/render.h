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
#include <akari/pmj02tables.h>
#include <akari/bluenoise.h>
#include <akari/image.h>
#include <akari/scenegraph.h>
#include <akari/render_xpu.h>
#include <array>
#include <memory>
namespace akari::scene {
    class SceneGraph;
}
namespace akari::render {
    template <class T>
    struct VarianceTracker {
        astd::optional<T> mean, m2;
        int count = 0;
        void update(T value) {
            if (count == 0) {
                mean = value;
                m2   = T(0.0);
            } else {
                auto delta = value - *mean;
                *mean += delta / T(count + 1);
                *m2 += delta * (value - *mean);
            }
            count++;
        }
        astd::optional<T> variance() const {
            if (count < 2) {
                return astd::nullopt;
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

    struct ImageTexture {
        std::shared_ptr<Image> image;
        ImageTexture() = default;
        ImageTexture(std::shared_ptr<Image> image) : image(std::move(image)) {}
        Float evaluate_f(const ShadingPoint &sp) const {

            vec2 texcoords = sp.texcoords;
            vec2 tc        = glm::mod(texcoords, vec2(1.0f));
            tc.y           = 1.0f - tc.y;
            return (*image)(tc, 0);
        }
        Spectrum evaluate_s(const ShadingPoint &sp) const {

            vec2 texcoords = sp.texcoords;
            vec2 tc        = glm::mod(texcoords, vec2(1.0f));
            tc.y           = 1.0f - tc.y;
            return Spectrum((*image)(tc, 0), (*image)(tc, 1), (*image)(tc, 2));
        }
    };
    template <>
    struct Texture<CPU> : Variant<ConstantTexture, ImageTexture> {
        using Variant<ConstantTexture, ImageTexture>::Variant;
        Float evaluate_f(const ShadingPoint &sp) const { AKR_VAR_DISPATCH(evaluate_f, sp); }
        Spectrum evaluate_s(const ShadingPoint &sp) const { AKR_VAR_DISPATCH(evaluate_s, sp); }
        Texture() : Texture(ConstantTexture(0.0)) {}
    };

    class MixBSDF {
      public:
        Float fraction;
        const BSDFClosure<CPU> *bsdf_A = nullptr;
        const BSDFClosure<CPU> *bsdf_B = nullptr;
        AKR_XPU MixBSDF(Float fraction, const BSDFClosure<CPU> *bsdf_A, const BSDFClosure<CPU> *bsdf_B)
            : fraction(fraction), bsdf_A(bsdf_A), bsdf_B(bsdf_B) {}
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const;
        [[nodiscard]] AKR_XPU BSDFValue evaluate(const Vec3 &wo, const Vec3 &wi) const;
        [[nodiscard]] AKR_XPU BSDFType type() const;
        AKR_XPU astd::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const;
        AKR_XPU BSDFValue albedo() const;
    };

    /*
    All BSDFClosure except MixBSDF must have *only* one of Diffuse, Glossy, Specular
    */
    template <>
    class BSDFClosure<CPU> : public Variant<DiffuseBSDF, MicrofacetReflection, SpecularReflection, SpecularTransmission,
                                            FresnelSpecular, MixBSDF> {
      public:
        using Variant::Variant;
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            AKR_VAR_DISPATCH(evaluate_pdf, wo, wi);
        }
        [[nodiscard]] AKR_XPU BSDFValue evaluate(const Vec3 &wo, const Vec3 &wi) const {
            AKR_VAR_DISPATCH(evaluate, wo, wi);
        }
        [[nodiscard]] AKR_XPU BSDFType type() const { AKR_VAR_DISPATCH(type); }
        [[nodiscard]] AKR_XPU bool match_flags(BSDFType flag) const { return ((uint32_t)type() & (uint32_t)flag) != 0; }
        [[nodiscard]] AKR_XPU astd::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            AKR_VAR_DISPATCH(sample, u, wo);
        }
        [[nodiscard]] AKR_XPU BSDFValue albedo() const { AKR_VAR_DISPATCH(albedo); }
    };

    AKR_XPU_CLASS class Material {
      public:
        Texture<C> color;
        Texture<C> metallic;
        Texture<C> roughness;
        Texture<C> specular;
        Texture<C> emission;
        Texture<C> transmission;
        AKR_XPU Material() {}
        BSDF<C> evaluate(Sampler<C> &sampler, Allocator<> alloc, const SurfaceInteraction<C> &si) const;
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
            *pdf    = pdfs[0] * pdfs[1];
            return Vec2(d0, d1);
        }
        Float pdf_continuous(const Vec2 &p) const {
            auto iu = std::clamp<int>(p[0] * pConditionalV[0].count(), 0, pConditionalV[0].count() - 1);
            auto iv = std::clamp<int>(p[1] * pMarginal->count(), 0, pMarginal->count() - 1);
            return pConditionalV[iv].func[iu] / pMarginal->funcInt;
        }
    };
#pragma endregion
#pragma endregion

    struct Film {
        Array2D<Spectrum> radiance;
        Array2D<Float> weight;
        Array2D<astd::array<AtomicFloat, Spectrum::size>> splats;
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
                        auto color  = (radiance(x, y)) / weight(x, y);
                        array(x, y) = color + splat_s;
                    } else {
                        auto color  = radiance(x, y);
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
                        auto color     = (radiance(x, y)) / weight(x, y) + splat_s;
                        image(x, y, 0) = color[0];
                        image(x, y, 1) = color[1];
                        image(x, y, 2) = color[2];
                    } else {
                        auto color     = radiance(x, y) + splat_s;
                        image(x, y, 0) = color[0];
                        image(x, y, 1) = color[1];
                        image(x, y, 2) = color[2];
                    }
                }
            });
            return image;
        }
    };

    struct Intersection {
        Float t = Inf;
        Vec2 uv;
        int geom_id = -1;
        int prim_id = -1;
        bool hit() const { return geom_id != -1; }
    };
    AKR_XPU_CLASS
    class Scene;
    class EmbreeAccel {
      public:
        virtual void build(const Scene<CPU> &scene, const std::shared_ptr<scene::SceneGraph> &scene_graph) = 0;
        virtual astd::optional<Intersection> intersect1(const Ray &ray) const                              = 0;
        virtual bool occlude1(const Ray &ray) const                                                        = 0;
        virtual Bounds3f world_bounds() const                                                              = 0;
    };
    std::shared_ptr<EmbreeAccel> create_embree_accel();

    AKR_XPU_CLASS
    class PowerLightSampler {
      public:
        PowerLightSampler(Allocator<> alloc, BufferView<const Light<C> *> lights_, const std::vector<Float> &power)
            : light_distribution(power.data(), power.size(), alloc), lights(lights_) {
            for (uint32_t i = 0; i < power.size(); i++) {
                light_pdf[lights[i]] = light_distribution.pdf_discrete(i);
            }
        }
        Distribution1D light_distribution;
        BufferView<const Light<C> *> lights;
        std::unordered_map<const Light<C> *, Float> light_pdf;
        std::pair<const Light<C> *, Float> sample(Vec2 u) const {
            auto [light_idx, pdf] = light_distribution.sample_discrete(u[0]);
            return std::make_pair(lights[light_idx], pdf);
        }
        Float pdf(const Light<C> *light) const {
            auto it = light_pdf.find(light);
            if (it == light_pdf.end()) {
                return 0.0;
            }
            return it->second;
        }
    };
    struct MLTSampler {
        struct PrimarySample {
            Float value;
            Float _backup;
            uint64_t last_modification_iteration;
            uint64_t last_modified_backup;

            void backup() {
                _backup              = value;
                last_modified_backup = last_modification_iteration;
            }

            void restore() {
                value                       = _backup;
                last_modification_iteration = last_modified_backup;
            }
        };
        AKR_XPU explicit MLTSampler(unsigned int seed) : rng(seed) {}
        Rng rng;
        std::vector<PrimarySample> X;
        uint64_t current_iteration = 0;
        bool large_step            = true;
        uint64_t last_large_step   = 0;
        Float large_step_prob      = 0.25;
        uint32_t sample_index      = 0;
        uint64_t accepts = 0, rejects = 0;
        AKR_XPU Float uniform() { return rng.uniform_float(); }
        AKR_XPU void start_next_sample() {
            AKR_CPU_ONLY {
                sample_index = 0;
                current_iteration++;
                large_step = uniform() < large_step_prob;
            }
        }
        AKR_XPU void set_sample_index(uint64_t idx) { AKR_PANIC("shouldn't be called"); }
        AKR_XPU Float next1d() {
            AKR_CPU_ONLY {
                if (sample_index >= X.size()) {
                    X.resize(sample_index + 1u);
                }
                auto &Xi = X[sample_index];
                mutate(Xi, sample_index);
                sample_index += 1;
                return Xi.value;
            }
        }
        AKR_XPU vec2 next2d() { return vec2(next1d(), next1d()); }
        AKR_XPU double mutate(double x, double s1, double s2) {
            double r = uniform();
            if (r < 0.5) {
                r = r * 2.0;
                x = x + s2 * std::exp(-std::log(s2 / s1) * r);
                if (x > 1.0)
                    x -= 1.0;
            } else {
                r = (r - 0.5) * 2.0;
                x = x - s2 * std::exp(-std::log(s2 / s1) * r);
                if (x < 0.0)
                    x += 1.0;
            }
            return x;
        }
        AKR_XPU void mutate(PrimarySample &Xi, int sampleIndex) {
            AKR_CPU_ONLY {
                double s1, s2;
                s1 = 1.0 / 1024.0, s2 = 1.0 / 64.0;

                if (Xi.last_modification_iteration < last_large_step) {
                    Xi.value                       = uniform();
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
                        Xi.value                       = x;
                        Xi.last_modification_iteration = current_iteration - 1;
                    }
                    Xi.backup();
                    Xi.value = mutate(Xi.value, s1, s2);
                }

                Xi.last_modification_iteration = current_iteration;
            }
        }
        AKR_XPU void accept() {
            if (large_step) {
                last_large_step = current_iteration;
            }
            accepts++;
        }

        AKR_XPU void reject() {
            AKR_CPU_ONLY {
                for (PrimarySample &Xi : X) {
                    if (Xi.last_modification_iteration == current_iteration) {
                        Xi.restore();
                    }
                }
                rejects++;
                --current_iteration;
            }
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
        vec2 next2d() { return vec2(next1d(), next1d()); }
        void start_next_sample() { idx = 0; }
        void set_sample_index(uint64_t) {}

      private:
        uint32_t idx = 0;
        Rng rng;
        astd::pmr::vector<Float> Xs;
    };
    template <>
    class Sampler<CPU> : public Variant<LCGSampler, PCGSampler, MLTSampler, ReplaySampler> {
      public:
        using Variant::Variant;
        Sampler() : Sampler(PCGSampler()) {}
        Float next1d() { AKR_VAR_DISPATCH(next1d); }
        vec2 next2d() { AKR_VAR_DISPATCH(next2d); }
        void start_next_sample() { AKR_VAR_DISPATCH(start_next_sample); }
        void set_sample_index(uint64_t idx) { AKR_VAR_DISPATCH(set_sample_index, idx); }
    };
    AKR_XPU_CLASS
    class Light : public Variant<AreaLight<C>> {
      public:
        using Variant<AreaLight<C>>::Variant;
        Spectrum Le(const Vec3 &wo, const ShadingPoint &sp) const { AKR_VAR_DISPATCH(Le, wo, sp); }
        Float pdf_incidence(const PointGeometry &ref, const vec3 &wi) const {
            AKR_VAR_DISPATCH(pdf_incidence, ref, wi);
        }
        template <class TSampler>
        LightRaySample sample_emission(TSampler &sampler) const {
            AKR_VAR_DISPATCH(sample_emission, sampler);
        }
        LightSample sample_incidence(const LightSampleContext &ctx) const { AKR_VAR_DISPATCH(sample_incidence, ctx); }
    };
    AKR_XPU_CLASS
    class LightSampler;
    template <>
    class LightSampler<CPU> : public Variant<std::shared_ptr<PowerLightSampler<CPU>>> {
      public:
        using Variant::Variant;
        std::pair<const Light<CPU> *, Float> sample(Vec2 u) const { AKR_VAR_PTR_DISPATCH(sample, u); }
        Float pdf(const Light<CPU> *light) const { AKR_VAR_PTR_DISPATCH(pdf, light); }
    };
    AKR_XPU_CLASS class Scene;
    template <>
    class Scene<CPU> {
      public:
        astd::optional<Camera<CPU>> camera;
        std::vector<MeshInstance<CPU>> instances;
        std::vector<const Material<CPU> *> materials;
        std::vector<const Light<CPU> *> lights;
        std::shared_ptr<EmbreeAccel> accel;
        Allocator<> allocator;
        astd::optional<LightSampler<CPU>> light_sampler;
        astd::pmr::monotonic_buffer_resource *rsrc;
        astd::optional<SurfaceInteraction<CPU>> intersect(const Ray &ray) const;
        bool occlude(const Ray &ray) const;
        Scene()              = default;
        Scene(const Scene &) = delete;
        ~Scene();
    };

    std::shared_ptr<const Scene<CPU>> create_scene(Allocator<>, const std::shared_ptr<scene::SceneGraph> &scene_graph);

    // experimental path space denoising
    struct PSDConfig {
        size_t filter_radius = 8;
    };
    struct PTConfig {
        Sampler<CPU> sampler;
        int min_depth = 3;
        int max_depth = 5;
        int spp       = 16;
    };
    Film render_pt(PTConfig config, const Scene<CPU> &scene);
    struct UPTConfig {
        Sampler<CPU> sampler;
        int min_depth = 3;
        int max_depth = 5;
        int spp       = 16;
    };
    Image render_unified(UPTConfig config, const Scene<CPU> &scene);
    Image render_pt_psd(PTConfig config, PSDConfig psd_config, const Scene<CPU> &scene);

    // separate emitter direct hit
    // useful for MLT
    std::pair<Spectrum, Spectrum> render_pt_pixel_separete_emitter_direct(PTConfig config, Allocator<>,
                                                                          const Scene<CPU> &scene,
                                                                          Sampler<CPU> &sampler, const vec2 &p_film);
    inline Spectrum render_pt_pixel_wo_emitter_direct(PTConfig config, Allocator<> allocator, const Scene<CPU> &scene,
                                                      Sampler<CPU> &sampler, const vec2 &p_film) {
        auto [_, rest] = render_pt_pixel_separete_emitter_direct(config, allocator, scene, sampler, p_film);
        return rest - _;
    }
    inline Spectrum render_pt_pixel(PTConfig config, Allocator<> allocator, const Scene<CPU> &scene,
                                    Sampler<CPU> &sampler, const vec2 &p_film) {
        auto [_, rest] = render_pt_pixel_separete_emitter_direct(config, allocator, scene, sampler, p_film);
        return rest;
    }

    struct IRConfig {
        Sampler<CPU> sampler;
        int min_depth = 3;
        int max_depth = 5;
        uint32_t spp  = 16;
    };

    // instant radiosity
    Image render_ir(IRConfig config, const Scene<CPU> &scene);

    struct SMSConfig {
        Sampler<CPU>  sampler;
        int min_depth = 3;
        int max_depth = 5;
        int spp       = 16;
    };

    // sms single scatter
    Film render_sms_ss(SMSConfig config, const Scene<CPU> &scene);
    struct BDPTConfig {
        Sampler<CPU> sampler;
        int min_depth = 3;
        int max_depth = 5;
        int spp       = 16;
    };

    Image render_bdpt(PTConfig config, const Scene<CPU> &scene);

    struct MLTConfig {
        int num_bootstrap = 100000;
        int num_chains    = 1024;
        int min_depth     = 3;
        int max_depth     = 5;
        int spp           = 16;
    };
    Image render_mlt(MLTConfig config, const Scene<CPU> &scene);
    Image render_smcmc(MLTConfig config, const Scene<CPU> &scene);
} // namespace akari::render
