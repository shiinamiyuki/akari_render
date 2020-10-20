

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
#include <akari/render/scenegraph.h>
#include <akari/shaders/common.h>
#include <akari/core/memory.h>
#include <akari/render/sampler.h>
#include <akari/render/interaction.h>
#include <optional>

namespace akari::render {
    enum BSDFType : int {
        BSDF_NONE = 0u,
        BSDF_REFLECTION = 1u << 0u,
        BSDF_TRANSMISSION = 1u << 1u,
        BSDF_DIFFUSE = 1u << 2u,
        BSDF_GLOSSY = 1u << 3u,
        // BSDF_SPECULAR = 1u << 4u,
        BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_REFLECTION | BSDF_TRANSMISSION,
    };
    struct BSDFSample {
        Vec3 wi = Vec3(0);
        Float pdf = 0.0;
        Spectrum f = Spectrum(0.0f);
        BSDFType sampled = BSDF_NONE;
    };

    struct BSDFSampleContext {
        const vec2 u1;
        const Vec3 wo;

        BSDFSampleContext(const vec2 &u1, const Vec3 &wo) : u1(u1), wo(wo) {}
    };

    class BSDFClosure {
      public:
        [[nodiscard]] virtual Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const = 0;
        [[nodiscard]] virtual Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const = 0;
        [[nodiscard]] virtual BSDFType type() const = 0;
        [[nodiscard]] virtual bool match_flags(BSDFType flag) const { return ((uint32_t)type() & (uint32_t)flag) != 0; }
        virtual Spectrum sample(const vec2 &u, const Vec3 &wo, Vec3 *wi, Float *pdf, BSDFType *sampledType) const = 0;
    };
    class BSDF {
        const BSDFClosure *closure_;
        Vec3 ng, ns;
        Frame frame;
        Float choice_pdf = 1.0f;

      public:
        BSDF() = default;
        explicit BSDF(const Vec3 &ng, const Vec3 &ns) : ng(ng), ns(ns) { frame = Frame(ns); }
        bool null() const { return closure() == nullptr; }
        void set_closure(const BSDFClosure *closure) { closure_ = closure; }
        void set_choice_pdf(Float pdf) { choice_pdf = pdf; }
        const BSDFClosure *closure() const { return closure_; }
        [[nodiscard]] Float evaluate_pdf(const Vec3 &wo, const Vec3 &wi) const {
            auto pdf = closure()->evaluate_pdf(frame.world_to_local(wo), frame.world_to_local(wi));
            return pdf * choice_pdf;
        }
        [[nodiscard]] Spectrum evaluate(const Vec3 &wo, const Vec3 &wi) const {
            auto f = closure()->evaluate(frame.world_to_local(wo), frame.world_to_local(wi));
            return f;
        }

        [[nodiscard]] BSDFType type() const { return closure()->type(); }
        [[nodiscard]] bool match_flags(BSDFType flag) const { return closure()->match_flags(flag); }
        BSDFSample sample(const BSDFSampleContext &ctx) const {
            auto wo = frame.world_to_local(ctx.wo);
            Vec3 wi;
            BSDFSample sample;
            sample.f = closure()->sample(ctx.u1, wo, &wi, &sample.pdf, &sample.sampled);
            sample.wi = frame.local_to_world(wi);
            sample.pdf *= choice_pdf;
            return sample;
        }
    };
    struct MaterialEvalContext {
        Allocator<> allocator;
        vec2 u1, u2;
        vec2 texcoords;
        Vec3 ng, ns;
        MaterialEvalContext(astd::pmr::monotonic_buffer_resource *rsrc) : allocator(rsrc) {}
        MaterialEvalContext(astd::pmr::monotonic_buffer_resource *rsrc, Sampler &sampler, const SurfaceInteraction &si)
            : MaterialEvalContext(rsrc, sampler, si.texcoords, si.ng, si.ns) {}
        MaterialEvalContext(astd::pmr::monotonic_buffer_resource *rsrc, Sampler &sampler, const vec2 &texcoords,
                            const Vec3 &ng, const Vec3 &ns)
            : allocator(rsrc), u1(sampler.next2d()), u2(sampler.next2d()), texcoords(texcoords), ng(ng), ns(ns) {}
    };

    class Material {
      public:
        virtual BSDF get_bsdf(MaterialEvalContext &ctx) const = 0;
    };

    class MaterialNode : public SceneNode {
      public:
        virtual Material *create_material(Allocator<> &allocator);
    };
} // namespace akari::render