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
#include <akari/mathutil.h>
#include <akari/util_xpu.h>
namespace akari::render {

    struct Rng {
        AKR_XPU Rng(uint64_t sequence = 0) { pcg32_init(sequence); }
        AKR_XPU uint32_t uniform_u32() { return pcg32(); }
        AKR_XPU double uniform_float() { return pcg32() / double(0xffffffff); }

      private:
        uint64_t state                   = 0x4d595df4d0f33173; // Or something seed-dependent
        static uint64_t const multiplier = 6364136223846793005u;
        static uint64_t const increment  = 1442695040888963407u; // Or an arbitrary odd constant
        static uint32_t rotr32(uint32_t x, unsigned r) { return x >> r | x << (-r & 31); }
        AKR_XPU uint32_t pcg32(void) {
            uint64_t x     = state;
            unsigned count = (unsigned)(x >> 59); // 59 = 64 - 5

            state = x * multiplier + increment;
            x ^= x >> 18;                              // 18 = (64 - 27)/2
            return rotr32((uint32_t)(x >> 27), count); // 27 = 32 - 5
        }
        AKR_XPU void pcg32_init(uint64_t seed) {
            state = seed + increment;
            (void)pcg32();
        }
    };
    // http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
    AKR_XPU inline uint64_t mix_bits(uint64_t v) {
        v ^= (v >> 31);
        v *= 0x7fb5d329728ea185;
        v ^= (v >> 27);
        v *= 0x81dadef4bc2dd44d;
        v ^= (v >> 33);
        return v;
    }
    AKR_XPU inline int permutation_element(uint32_t i, uint32_t l, uint32_t p) {
        uint32_t w = l - 1;
        w |= w >> 1;
        w |= w >> 2;
        w |= w >> 4;
        w |= w >> 8;
        w |= w >> 16;
        do {
            i ^= p;
            i *= 0xe170893d;
            i ^= p >> 16;
            i ^= (i & w) >> 4;
            i ^= p >> 8;
            i *= 0x0929eb3f;
            i ^= p >> 23;
            i ^= (i & w) >> 1;
            i *= 1 | p >> 27;
            i *= 0x6935fa69;
            i ^= (i & w) >> 11;
            i *= 0x74dcb303;
            i ^= (i & w) >> 2;
            i *= 0x9e501cc3;
            i ^= (i & w) >> 2;
            i *= 0xc860a3df;
            i &= w;
            i ^= i >> 5;
        } while (i >= l);
        return (i + p) % l;
    }
    struct SamplerConfig {
        enum Type {
            PCG,
            LCG,
            PMJ02BN,
        };
        Type type           = Type::PCG;
        int pixel_tile_size = 16;
        int spp             = 16;
    };
    class PMJ02BNSampler {
        int spp = 0;
        int seed;
        int dimension = 0, sample_index = 0;
        ivec2 pixel;
        std::shared_ptr<vec2[]> pixel_samples;
        int pixel_tile_size = 16;

      public:
        AKR_XPU void start_pixel_sample(ivec2 p, uint32_t idx, uint32_t dim) {
            pixel        = p;
            sample_index = idx;
            dimension    = std::max(2u, dim);
        }
        AKR_XPU Float next1d() {
            uint64_t hash =
                mix_bits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^ ((uint64_t)dimension << 16) ^ seed);
            int index   = permutation_element(sample_index, spp, hash);
            Float delta = blue_nosie(dimension, pixel);
            ++dimension;
            return std::min((index + delta) / spp, OneMinusEpsilon);
        }
        AKR_XPU vec2 next2d() {
            if (dimension == 0) {
                // Return pmj02bn pixel sample
                int px = pixel.x % pixel_tile_size, py = pixel.y % pixel_tile_size;
                int offset = (px + py * pixel_tile_size) * spp;
                dimension += 2;
                return (pixel_samples.get())[offset + sample_index];

            } else {
                // Compute index for 2D pmj02bn sample
                int index       = sample_index;
                int pmjInstance = dimension / 2;
                if (pmjInstance >= N_PMJ02BN_SETS) {
                    // Permute index to be used for pmj02bn sample array
                    uint64_t hash = mix_bits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                             ((uint64_t)dimension << 16) ^ seed);
                    index         = permutation_element(sample_index, spp, hash);
                }

                // Return randomized pmj02bn sample for current dimension
                auto u = pmj02bn(pmjInstance, index);
                // Apply Cranley-Patterson rotation to pmj02bn sample _u_
                u += vec2(blue_nosie(dimension, pixel), blue_nosie(dimension + 1, pixel));
                if (u.x >= 1)
                    u.x -= 1;
                if (u.y >= 1)
                    u.y -= 1;

                dimension += 2;
                return {std::min(u.x, OneMinusEpsilon), std::min(u.y, OneMinusEpsilon)};
            }
        }
        void start_next_sample() {}
    };

    class PCGSampler {
        Rng rng;

      public:
        AKR_XPU void set_sample_index(uint64_t idx) { rng = Rng(idx); }
        AKR_XPU Float next1d() { return rng.uniform_float(); }
        AKR_XPU vec2 next2d() { return vec2(next1d(), next1d()); }
        AKR_XPU void start_next_sample() {}
        AKR_XPU PCGSampler(uint64_t seed = 0u) : rng(seed) {}
    };
    class LCGSampler {
        uint32_t seed;

      public:
        AKR_XPU void set_sample_index(uint64_t idx) { seed = idx & 0xffffffff; }
        AKR_XPU Float next1d() {
            seed = (1103515245 * seed + 12345);
            return (Float)seed / (Float)0xFFFFFFFF;
        }
        AKR_XPU vec2 next2d() { return vec2(next1d(), next1d()); }
        AKR_XPU void start_next_sample() {}
        AKR_XPU LCGSampler(uint64_t seed = 0u) : seed(seed) {}
    };

    class GPUSampler : public Variant<LCGSampler, PCGSampler> {
      public:
        using Variant::Variant;
        GPUSampler() : GPUSampler(PCGSampler()) {}
        AKR_XPU Float next1d() { AKR_VAR_DISPATCH(next1d); }
        AKR_XPU vec2 next2d() { AKR_VAR_DISPATCH(next2d); }
        AKR_XPU void start_next_sample() { AKR_VAR_DISPATCH(start_next_sample); }
        AKR_XPU void set_sample_index(uint64_t idx) { AKR_VAR_DISPATCH(set_sample_index, idx); }
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
        Float lens_radius    = 0.0f;
        Float focal_distance = 0.0f;
        AKR_XPU PerspectiveCamera(const ivec2 &_resolution, const Transform &c2w, Float fov)
            : c2w(c2w), w2c(c2w.inverse()), _resolution(_resolution), fov(fov) {
            preprocess();
        }
        AKR_XPU ivec2 resolution() const { return _resolution; }
        AKR_XPU CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
            CameraSample sample;
            sample.p_lens = concentric_disk_sampling(u1) * lens_radius;
            sample.p_film = vec2(raster) + u2;
            sample.weight = 1;

            vec2 p = shuffle<0, 1>(r2c.apply_point(Vec3(sample.p_film.x, sample.p_film.y, 0.0f)));
            Ray ray(Vec3(0), Vec3(normalize(Vec3(p.x, p.y, 0) - Vec3(0, 0, 1))));
            if (lens_radius > 0 && focal_distance > 0) {
                Float ft    = focal_distance / std::abs(ray.d.z);
                Vec3 pFocus = ray(ft);
                ray.o       = Vec3(sample.p_lens.x, sample.p_lens.y, 0);
                ray.d       = Vec3(normalize(pFocus - ray.o));
            }
            ray.o         = c2w.apply_point(ray.o);
            ray.d         = c2w.apply_vector(ray.d);
            sample.normal = c2w.apply_normal(Vec3(0, 0, -1.0f));
            sample.ray    = ray;

            return sample;
        }

      private:
        AKR_XPU void preprocess() {
            Transform m;
            m      = Transform::scale(Vec3(1.0f / _resolution.x, 1.0f / _resolution.y, 1)) * m;
            m      = Transform::scale(Vec3(2, 2, 1)) * m;
            m      = Transform::translate(Vec3(-1, -1, 0)) * m;
            m      = Transform::scale(Vec3(1, -1, 1)) * m;
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
        AKR_XPU ivec2 resolution() const { AKR_VAR_DISPATCH(resolution); }
        AKR_XPU CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
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
        AKR_XPU ShadingPoint(Vec2 tc) : texcoords(tc) {}
    };

    struct ConstantTexture {
        Spectrum value;
        AKR_XPU ConstantTexture(Float v) : value(v) {}
        AKR_XPU ConstantTexture(Spectrum v) : value(v) {}
        AKR_XPU Float evaluate_f(const ShadingPoint &sp) const { return value[0]; }
        AKR_XPU Spectrum evaluate_s(const ShadingPoint &sp) const { return value; }
    };

    struct GPUImageTexture {};

    struct GPUTexture : Variant<ConstantTexture> {
        using Variant::Variant;
        AKR_XPU Float evaluate_f(const ShadingPoint &sp) const { AKR_VAR_DISPATCH(evaluate_f, sp); }
        AKR_XPU Spectrum evaluate_s(const ShadingPoint &sp) const { AKR_VAR_DISPATCH(evaluate_s, sp); }
        AKR_XPU GPUTexture() : GPUTexture(ConstantTexture(0.0)) {}
    };

    enum class BSDFType : int {
        Unset                = 0u,
        Reflection           = 1u << 0,
        Transmission         = 1u << 1,
        Diffuse              = 1u << 2,
        Glossy               = 1u << 3,
        Specular             = 1u << 4,
        DiffuseReflection    = Diffuse | Reflection,
        DiffuseTransmission  = Diffuse | Transmission,
        GlossyReflection     = Glossy | Reflection,
        GlossyTransmission   = Glossy | Transmission,
        SpecularReflection   = Specular | Reflection,
        SpecularTransmission = Specular | Transmission,
        All                  = Diffuse | Glossy | Specular | Reflection | Transmission
    };
    AKR_XPU inline BSDFType operator&(BSDFType a, BSDFType b) { return BSDFType((int)a & (int)b); }
    AKR_XPU inline BSDFType operator|(BSDFType a, BSDFType b) { return BSDFType((int)a | (int)b); }
    AKR_XPU inline BSDFType operator~(BSDFType a) { return BSDFType(~(uint32_t)a); }

    struct BSDFValue {
        Spectrum diffuse;
        Spectrum glossy;
        Spectrum specular;
        AKR_XPU static BSDFValue zero() { return BSDFValue{Spectrum(0.0), Spectrum(0.0), Spectrum(0.0)}; }
        AKR_XPU static BSDFValue with_diffuse(Spectrum diffuse) {
            return BSDFValue{diffuse, Spectrum(0.0), Spectrum(0.0)};
        }
        AKR_XPU static BSDFValue with_glossy(Spectrum glossy) {
            return BSDFValue{Spectrum(0.0), glossy, Spectrum(0.0)};
        }
        AKR_XPU static BSDFValue with_specular(Spectrum specular) {
            return BSDFValue{Spectrum(0.0), Spectrum(0.0), specular};
        }
        // linear interpolation
        AKR_XPU static BSDFValue mix(Float alpha, const BSDFValue &x, const BSDFValue &y) {
            return BSDFValue{(1.0f - alpha) * x.diffuse + alpha * y.diffuse,
                             (1.0f - alpha) * x.glossy + alpha * y.glossy,
                             (1.0f - alpha) * x.specular + alpha * y.specular};
        }
        AKR_XPU BSDFValue operator*(Float k) const { return BSDFValue{diffuse * k, glossy * k, specular * k}; }
        AKR_XPU friend BSDFValue operator*(Float k, const BSDFValue &f) { return f * k; }
        AKR_XPU BSDFValue operator*(const Spectrum &k) const {
            return BSDFValue{diffuse * k, glossy * k, specular * k};
        }
        AKR_XPU friend BSDFValue operator*(const Spectrum &k, const BSDFValue &f) { return f * k; }
        AKR_XPU BSDFValue operator+(const BSDFValue &rhs) const {
            return BSDFValue{diffuse + rhs.diffuse, glossy + rhs.glossy, specular + rhs.specular};
        }
        AKR_XPU Spectrum operator()() const { return diffuse + glossy + specular; }
    };

    struct BSDFSample {
        Vec3 wi;
        BSDFValue f   = BSDFValue::zero();
        Float pdf     = 0.0;
        BSDFType type = BSDFType::Unset;
    };
    class BSDFClosure;
    class DiffuseBSDF {
        Spectrum R;

      public:
        DiffuseBSDF(const Spectrum &R) : R(R) {}
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {

            if (same_hemisphere(wo, wi)) {
                return cosine_hemisphere_pdf(std::abs(cos_theta(wi)));
            }
            return 0.0f;
        }
        [[nodiscard]] AKR_XPU BSDFValue evaluate(const Vec3 &wo, const Vec3 &wi) const {

            if (same_hemisphere(wo, wi)) {
                return BSDFValue::with_diffuse(R * InvPi);
            }
            return BSDFValue::with_diffuse(Spectrum(0.0f));
        }
        [[nodiscard]] AKR_XPU BSDFType type() const { return BSDFType::DiffuseReflection; }
        AKR_XPU astd::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            sample.wi = cosine_hemisphere_sampling(u);
            if (!same_hemisphere(wo, sample.wi)) {
                sample.wi.y = -sample.wi.y;
            }
            sample.type = type();
            sample.pdf  = cosine_hemisphere_pdf(std::abs(cos_theta(sample.wi)));
            sample.f    = BSDFValue::with_diffuse(R * InvPi);
            return sample;
        }
        [[nodiscard]] AKR_XPU BSDFValue albedo() const { return BSDFValue::with_diffuse(R); }
    };

    class MicrofacetReflection {
      public:
        Spectrum R;
        MicrofacetModel model;
        Float roughness;
        MicrofacetReflection(const Spectrum &R, Float roughness)
            : R(R), model(microfacet_new(MicrofacetGGX, roughness)), roughness(roughness) {}
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            if (same_hemisphere(wo, wi)) {
                auto wh = normalize(wo + wi);
                return microfacet_evaluate_pdf(model, wh) / (Float(4.0f) * dot(wo, wh));
            }
            return 0.0f;
        }
        [[nodiscard]] AKR_XPU BSDFValue evaluate(const Vec3 &wo, const Vec3 &wi) const {
            if (same_hemisphere(wo, wi)) {
                Float cosThetaO = abs_cos_theta(wo);
                Float cosThetaI = abs_cos_theta(wi);
                auto wh         = (wo + wi);
                if (cosThetaI == 0 || cosThetaO == 0)
                    return BSDFValue::zero();
                if (wh.x == 0 && wh.y == 0 && wh.z == 0)
                    return BSDFValue::zero();
                wh = normalize(wh);
                if (wh.y < 0) {
                    wh = -wh;
                }
                auto F = 1.0f; // fresnel->evaluate(dot(wi, wh));

                return BSDFValue::with_glossy(R * (microfacet_D(model, wh) * microfacet_G(model, wo, wi, wh) * F /
                                                   (Float(4.0f) * cosThetaI * cosThetaO)));
            }
            return BSDFValue::zero();
        }
        [[nodiscard]] AKR_XPU BSDFType type() const { return BSDFType::GlossyReflection; }
        [[nodiscard]] AKR_XPU astd::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            sample.type = type();
            auto wh     = microfacet_sample_wh(model, wo, u);
            sample.wi   = glm::reflect(-wo, wh);
            if (!same_hemisphere(wo, sample.wi)) {
                sample.pdf = 0;
                return astd::nullopt;
            } else {
                if (wh.y < 0) {
                    wh = -wh;
                }
                sample.pdf = microfacet_evaluate_pdf(model, wh) / (Float(4.0f) * abs(dot(wo, wh)));
                AKR_ASSERT(sample.pdf >= 0.0);
            }
            sample.f = evaluate(wo, sample.wi);
            return sample;
        }
        [[nodiscard]] AKR_XPU BSDFValue albedo() const { return BSDFValue::with_glossy(R); }
    };

    class FresnelNoOp {
      public:
        [[nodiscard]] AKR_XPU Spectrum evaluate(Float cosThetaI) const;
    };

    class FresnelConductor {
        const Spectrum etaI, etaT, k;

      public:
        AKR_XPU FresnelConductor(const Spectrum &etaI, const Spectrum &etaT, const Spectrum &k)
            : etaI(etaI), etaT(etaT), k(k) {}
        [[nodiscard]] AKR_XPU Spectrum evaluate(Float cosThetaI) const;
    };
    class FresnelDielectric {
        Float etaI, etaT;

      public:
        AKR_XPU FresnelDielectric(const Float &etaI, const Float &etaT) : etaI(etaI), etaT(etaT) {}
        [[nodiscard]] AKR_XPU Spectrum evaluate(Float cosThetaI) const;
    };
    class Fresnel : public Variant<FresnelConductor, FresnelDielectric, FresnelNoOp> {
      public:
        using Variant::Variant;
        AKR_XPU Fresnel() : Variant(FresnelNoOp()) {}
        [[nodiscard]] AKR_XPU Spectrum evaluate(Float cosThetaI) const { AKR_VAR_DISPATCH(evaluate, cosThetaI); }
    };
    class SpecularReflection {
        Spectrum R;

      public:
        AKR_XPU SpecularReflection(const Spectrum &R) : R(R) {}
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.0f; }
        [[nodiscard]] AKR_XPU BSDFValue evaluate(const Vec3 &wo, const Vec3 &wi) const { return BSDFValue::zero(); }
        [[nodiscard]] AKR_XPU BSDFType type() const { return BSDFType::SpecularReflection; }
        AKR_XPU astd::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            sample.wi   = glm::reflect(-wo, vec3(0, 1, 0));
            sample.type = type();
            sample.pdf  = 1.0;
            sample.f    = BSDFValue::with_specular(R / (std::abs(cos_theta(sample.wi))));
            return sample;
        }
        [[nodiscard]] AKR_XPU BSDFValue albedo() const { return BSDFValue::with_specular(R); }
    };
    class SpecularTransmission {
        Spectrum R;
        Float eta;

      public:
        AKR_XPU SpecularTransmission(const Spectrum &R, Float eta) : R(R), eta(eta) {}
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const { return 0.0f; }
        [[nodiscard]] AKR_XPU BSDFValue evaluate(const Vec3 &wo, const Vec3 &wi) const { return BSDFValue::zero(); }
        [[nodiscard]] AKR_XPU BSDFType type() const { return BSDFType::SpecularTransmission; }
        AKR_XPU astd::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const {
            BSDFSample sample;
            Float etaIO = same_hemisphere(wo, vec3(0, 1, 0)) ? eta : 1.0f / eta;
            auto wt     = refract(wo, faceforward(wo, vec3(0, 1, 0)), etaIO);
            if (glm::all(glm::equal(wt, vec3(0)))) {
                return astd::nullopt;
            }
            sample.wi   = wt;
            sample.type = type();
            sample.pdf  = 1.0;
            sample.f    = BSDFValue::with_specular(R / (std::abs(cos_theta(sample.wi))));
            return sample;
        }
        [[nodiscard]] BSDFValue albedo() const { return BSDFValue::with_specular(R); }
    };

    class FresnelSpecular {
        Spectrum R, T;
        Float etaA, etaB;
        FresnelDielectric fresnel;

      public:
        AKR_XPU explicit FresnelSpecular(const Spectrum &R, const Spectrum &T, Float etaA, Float etaB)
            : R(R), T(T), etaA(etaA), etaB(etaB), fresnel(etaA, etaB) {}
        [[nodiscard]] AKR_XPU BSDFType type() const {
            return BSDFType::SpecularTransmission | BSDFType::SpecularReflection;
        }
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const vec3 &wo, const vec3 &wi) const { return 0; }
        [[nodiscard]] AKR_XPU BSDFValue evaluate(const vec3 &wo, const vec3 &wi) const { return BSDFValue::zero(); }
        [[nodiscard]] AKR_XPU astd::optional<BSDFSample> sample(const vec2 &u, const Vec3 &wo) const;
        [[nodiscard]] AKR_XPU BSDFValue albedo() const { return BSDFValue::with_specular((R + T) * 0.5); }
    };

    struct BSDFSampleContext {
        Float u0;
        Vec2 u1;
        const Vec3 wo;
    };
    /*
     All BSDFClosure except MixBSDF must have *only* one of Diffuse, Glossy, Specular
   */
    class GPUBSDFClosure
        : public Variant<DiffuseBSDF, MicrofacetReflection, SpecularReflection, SpecularTransmission, FresnelSpecular> {
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

    template <class TBSDFClosure>
    class TBSDF {
        astd::optional<BSDFClosure> closure_;
        Frame frame;
        Float choice_pdf = 1.0f;

      public:
        AKR_XPU BSDF(const Frame &frame) : frame(frame) {}
        AKR_XPU bool null() const { return !closure_.has_value(); }
        AKR_XPU void set_closure(const BSDFClosure &closure) { closure_ = closure; }
        AKR_XPU void set_choice_pdf(Float pdf) { choice_pdf = pdf; }
        AKR_XPU const BSDFClosure &closure() const { return *closure_; }
        [[nodiscard]] AKR_XPU Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            auto pdf = closure().evaluate_pdf(frame.world_to_local(wo), frame.world_to_local(wi));
            return pdf * choice_pdf;
        }
        [[nodiscard]] AKR_XPU BSDFValue evaluate(const Vec3 &wo, const Vec3 &wi) const {
            auto f = closure().evaluate(frame.world_to_local(wo), frame.world_to_local(wi));
            return f;
        }

        [[nodiscard]] AKR_XPU BSDFType type() const { return closure().type(); }
        [[nodiscard]] AKR_XPU bool is_pure_delta() const {
            auto ty = type();
            if (BSDFType::Unset == (ty & BSDFType::Specular))
                return false;
            if (BSDFType::Unset != (ty & BSDFType::Diffuse))
                return false;
            if (BSDFType::Unset != (ty & BSDFType::Glossy))
                return false;
            return true;
        }
        [[nodiscard]] AKR_XPU bool match_flags(BSDFType flag) const { return closure().match_flags(flag); }
        AKR_XPU astd::optional<BSDFSample> sample(const BSDFSampleContext &ctx) const {
            auto wo = frame.world_to_local(ctx.wo);
            if (auto sample = closure().sample(ctx.u1, wo)) {
                sample->wi = frame.local_to_world(sample->wi);
                sample->pdf *= choice_pdf;
                return sample;
            }
            return astd::nullopt;
        }
    };
    using GPUBSDF = TBSDF<GPUBSDFClosure>;
    struct Light;
    struct Material;
    struct GPUMaterial;
    struct Medium;
    struct Triangle {
        astd::array<Vec3, 3> vertices;
        astd::array<Vec3, 3> normals;
        astd::array<vec2, 3> texcoords;
        const Material *material        = nullptr;
        const GPUMaterial *gpu_material = nullptr;
        const Light *light              = nullptr;
        AKR_XPU Vec3 p(const vec2 &uv) const { return lerp3(vertices[0], vertices[1], vertices[2], uv); }
        AKR_XPU Float area() const {
            return length(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) * 0.5f;
        }
        AKR_XPU Vec3 ng() const { return normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])); }
        AKR_XPU Vec3 ns(const vec2 &uv) const { return normalize(lerp3(normals[0], normals[1], normals[2], uv)); }
        AKR_XPU vec2 texcoord(const vec2 &uv) const { return lerp3(texcoords[0], texcoords[1], texcoords[2], uv); }
        AKR_XPU Vec3 dpdu(Float u) const { return dlerp3du(vertices[0], vertices[1], vertices[2], u); }
        AKR_XPU Vec3 dpdv(Float v) const { return dlerp3du(vertices[0], vertices[1], vertices[2], v); }

        AKR_XPU std::pair<Vec3, Vec3> dnduv(const vec2 &uv) const {
            auto n   = ns(uv);
            Float il = 1.0 / length(n);
            n *= il;
            auto dn_du = (normals[1] - normals[0]) * il;
            auto dn_dv = (normals[2] - normals[0]) * il;
            dn_du      = -n * dot(n, dn_du) + dn_du;
            dn_dv      = -n * dot(n, dn_dv) + dn_dv;
            return std::make_pair(dn_du, dn_dv);
        }

        AKR_XPU astd::optional<std::pair<Float, Vec2>> intersect(const Ray &ray) const {
            auto &v0 = vertices[0];
            auto &v1 = vertices[1];
            auto &v2 = vertices[2];
            Vec3 e1  = (v1 - v0);
            Vec3 e2  = (v2 - v0);
            Float a, f, u, v;
            auto h = cross(ray.d, e2);
            a      = dot(e1, h);
            if (a > Float(-1e-6f) && a < Float(1e-6f))
                return astd::nullopt;
            f      = 1.0f / a;
            auto s = ray.o - v0;
            u      = f * dot(s, h);
            if (u < 0.0 || u > 1.0)
                return astd::nullopt;
            auto q = cross(s, e1);
            v      = f * dot(ray.d, q);
            if (v < 0.0 || u + v > 1.0)
                return astd::nullopt;
            Float t = f * dot(e2, q);
            if (t > ray.tmin && t < ray.tmax) {
                return std::make_pair(t, Vec2(u, v));
            } else {
                return astd::nullopt;
            }
        }
    };

    AKR_XPU inline Float phase_hg(Float cosTheta, Float g) {
        Float denom = 1 + g * g + 2 * g * cosTheta;
        return Inv4Pi * (1 - g * g) / (denom * std::sqrt(denom));
    }

    struct PhaseHG {
        const Float g;
        AKR_XPU inline Float evaluate(const Float cos_theta) const { return phase_hg(cos_theta, g); }
        AKR_XPU std::pair<Vec3, Float> sample(const Vec3 &wo, const Vec2 &u) const {
            Float cos_theta = 0.0;
            if (std::abs(g) < 1e-3)
                cos_theta = 1 - 2 * u[0];
            else {
                Float sqr = (1 - g * g) / (1 + g - 2 * g * u[0]);
                cos_theta = -(1 + g * g - sqr * sqr) / (2 * g);
            }
            auto sin_theta = std::sqrt(std::max<Float>(0, 1.0 - cos_theta * cos_theta));
            auto phi       = 2.0 * Pi * u[1];
            Frame frame(wo);
            auto wi = spherical_to_xyz(sin_theta, cos_theta, phi);
            return std::make_pair(frame.local_to_world(wi), evaluate(cos_theta));
        }
    };
    struct PhaseFunction : Variant<PhaseHG> {
        using Variant::Variant;
        AKR_XPU Float evaluate(Float cos_theta) const { AKR_VAR_DISPATCH(evaluate, cos_theta); }
        AKR_XPU std::pair<Vec3, Float> sample(const Vec3 &wo, const Vec2 &u) const { AKR_VAR_DISPATCH(sample, wo, u); }
    };
    struct MediumInteraction {
        Vec3 p;
        PhaseFunction phase;
    };
    struct HomogeneousMedium {
        const Spectrum sigma_a, sigma_s;
        const Spectrum sigma_t;
        const Float g;
        AKR_XPU HomogeneousMedium(Spectrum sigma_a, Spectrum sigma_s, Float g)
            : sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_a + sigma_s), g(g) {}
        template <class TSampler>
        Spectrum transmittance(const Ray &ray, TSampler &sampler) const {
            return exp(-sigma_t * std::min<Float>(ray.tmax * length(ray.d), MaxFloat));
        }
        template <class TSampler>
        AKR_XPU std::pair<astd::optional<MediumInteraction>, Spectrum> sample(const Ray &ray, TSampler &sampler,
                                                                              Allocator<> alloc) const {
            int channel        = std::min<int>(sampler.next1d() * Spectrum::size, Spectrum::size - 1);
            auto dist          = -std::log(1.0 - sampler.next1d()) / sigma_t[channel];
            auto t             = std::min<double>(dist * length(ray.d), ray.tmax);
            bool sample_medium = t < ray.tmax;
            astd::optional<MediumInteraction> mi;
            if (sample_medium) {
                mi.emplace(MediumInteraction{ray(t), PhaseHG{g}});
            }
            auto tr          = transmittance(ray, sampler);
            Spectrum density = sample_medium ? sigma_t * tr : tr;
            Float pdf        = hsum(density);
            pdf /= Spectrum::size;
            return std::make_pair(mi, Spectrum(sample_medium ? (tr * sigma_s / pdf) : (tr / pdf)));
        }
    };
    struct Medium : Variant<HomogeneousMedium> {
        using Variant::Variant;
        template <class TSampler>
        AKR_XPU Spectrum transmittance(const Ray &ray, TSampler &sampler) const {
            AKR_VAR_DISPATCH(transmittance, ray, sampler);
        }
        template <class TSampler>
        AKR_XPU std::pair<astd::optional<MediumInteraction>, Spectrum> sample(const Ray &ray, TSampler &sampler,
                                                                              Allocator<> alloc) const {
            AKR_VAR_DISPATCH(sample, ray, sampler, alloc);
        }
    };
    struct Material;
    struct GPUMaterial;
    struct MeshInstance {
        Transform transform;
        BufferView<const vec3> vertices;
        BufferView<const uvec3> indices;
        BufferView<const vec3> normals;
        BufferView<const vec2> texcoords;
        std::vector<const Light *> lights;
        const scene::Mesh *mesh         = nullptr;
        const Material *material        = nullptr;
        const GPUMaterial *gpu_material = nullptr;
        const Medium *medium            = nullptr;

        AKR_XPU Triangle get_triangle(int prim_id) const {
            Triangle trig;
            for (int i = 0; i < 3; i++) {
                trig.vertices[i] = transform.apply_vector(vertices[indices[prim_id][i]]);
                trig.normals[i]  = transform.apply_normal(normals[indices[prim_id][i]]);
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
    struct SurfaceInteraction {
        Triangle triangle;
        Vec3 p;
        Vec3 ng, ns;
        vec2 texcoords;
        Vec3 dndu, dndv;
        Vec3 dpdu, dpdv;
        const MeshInstance *shape = nullptr;
        AKR_XPU SurfaceInteraction(const vec2 &uv, const Triangle &triangle)
            : triangle(triangle), p(triangle.p(uv)), ng(triangle.ng()), ns(triangle.ns(uv)),
              texcoords(triangle.texcoord(uv)) {
            dpdu                 = triangle.dpdu(uv[0]);
            dpdv                 = triangle.dpdu(uv[1]);
            std::tie(dndu, dndv) = triangle.dnduv(uv);
        }
        AKR_XPU const Light *light() const { return triangle.light; }
        AKR_XPU const Material *material() const { return triangle.material; }
        AKR_XPU const Medium *medium() const { return shape->medium; }
        AKR_XPU ShadingPoint sp() const {
            ShadingPoint sp_;
            sp_.n         = ns;
            sp_.texcoords = texcoords;
            sp_.dndu      = dndu;
            sp_.dndv      = dndv;
            sp_.dpdu      = dpdu;
            sp_.dpdv      = dpdv;
            return sp_;
        }
    };

    struct PointGeometry {
        Vec3 p;
        Vec3 n;
    };

    struct LightSampleContext {
        vec2 u;
        Vec3 p;
        Vec3 n = Vec3(0);
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
    template <class TTexture>
    struct AreaLight {
        Triangle triangle;
        TTexture color;
        bool double_sided = false;
        AKR_XPU AreaLight(Triangle triangle, TTexture color, bool double_sided)
            : triangle(triangle), color(color), double_sided(double_sided) {
            ng = triangle.ng();
        }
        AKR_XPU Spectrum Le(const Vec3 &wo, const ShadingPoint &sp) const {
            bool face_front = dot(wo, ng) > 0.0;
            if (double_sided || face_front) {
                return color.evaluate_s(sp);
            }
            return Spectrum(0.0);
        }
        AKR_XPU Float pdf_incidence(const PointGeometry &ref, const vec3 &wi) const {
            Ray ray(ref.p, wi);
            auto hit = triangle.intersect(ray);
            if (!hit) {
                return 0.0f;
            }
            Float SA = triangle.area() * (-glm::dot(wi, triangle.ng())) / (hit->first * hit->first);
            return 1.0f / SA;
        }
        template <class TSampler>
        AKR_XPU LightRaySample sample_emission(TSampler &sampler) const {
            LightRaySample sample;
            sample.uv   = sampler.next2d();
            auto coords = uniform_sample_triangle(sample.uv);
            auto p      = triangle.p(coords);

            sample.ng     = triangle.ng();
            sample.pdfPos = 1.0 / triangle.area();
            auto w        = cosine_hemisphere_sampling(sampler.next2d());
            Frame local(sample.ng);
            sample.pdfDir = cosine_hemisphere_pdf(std::abs(w.y));
            sample.ray    = Ray(p, local.local_to_world(w));
            sample.E      = color.evaluate_s(ShadingPoint(triangle.texcoord(coords)));
            return sample;
        }
        AKR_XPU LightSample sample_incidence(const LightSampleContext &ctx) const {
            auto coords = uniform_sample_triangle(ctx.u);
            auto p      = triangle.p(coords);
            LightSample sample;
            sample.ng     = triangle.ng();
            sample.wi     = p - ctx.p;
            auto dist_sqr = dot(sample.wi, sample.wi);
            sample.wi /= sqrt(dist_sqr);
            sample.I       = color.evaluate_s(ShadingPoint(triangle.texcoord(coords)));
            auto cos_theta = dot(sample.wi, sample.ng);
            if (-cos_theta < 0.0)
                sample.pdf = 0.0;
            else
                sample.pdf = dist_sqr / max(Float(0.0), -cos_theta) / triangle.area();
            // sample.shadow_ray = Ray(p, -sample.wi, Eps / std::abs(dot(sample.wi, sample.ng)),
            // sqrt(dist_sqr) * (Float(1.0f) - ShadowEps));
            sample.shadow_ray = Ray(ctx.p, sample.wi, Eps, sqrt(dist_sqr) * (Float(1.0f) - ShadowEps));
            return sample;
        }

      private:
        Vec3 ng;
    };

} // namespace akari::render