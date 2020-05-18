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

#include <akari/core/plugin.h>
#include <akari/render/geometry.hpp>
#include <akari/render/material.h>
#include <akari/render/reflection.h>
#include <akari/render/microfacet.h>
#include <akari/render/texture.h>
#include <utility>

namespace akari {
    class GlassMaterial final : public Material {

        [[refl]] std::shared_ptr<Texture> color;
        [[refl]] std::shared_ptr<Texture> ior;
        [[refl]] std::shared_ptr<Texture> roughness;

      public:
        GlassMaterial() = default;
        explicit GlassMaterial(std::shared_ptr<Texture> color) : color(std::move(color)) {}
        AKR_IMPLS(Material, Component)
        void compute_scattering_functions(SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,
                                          Float scale) const override {
            auto c = color->evaluate(si->sp) * scale;
            auto eta = ior->evaluate(si->sp)[0];
            auto alpha = roughness ? roughness->evaluate(si->sp)[0] : 0.000;
            alpha *= alpha;
            if (alpha == 0) {
                si->bsdf->add_component(arena.alloc<FresnelSpecular>(c, c, 1.0f, eta, mode));
            } else {
                 si->bsdf->add_component(arena.alloc<FresnelGlossy>(c, c,  MicrofacetModel(EGGX, alpha), 1.0f, eta, mode));
                // auto fresnel = arena.alloc<FresnelDielectric>(1.0f, eta);
                // auto fresnel = arena.alloc<FresnelNoOp>();
                // si->bsdf->add_component(arena.alloc<SpecularReflection>(c, fresnel));
                // si->bsdf->add_component(
                    // arena.alloc<MicrofacetTransmission>(c, MicrofacetModel(EGGX, alpha), 1.0f, eta, mode));
                // si->bsdf->add_component(arena.alloc<SpecularTransmission>(c,1.0f, eta, mode));
            }
        }
        bool support_bidirectional() const override { return true; }
    };

#include "generated/GlassMaterial.hpp"

    AKR_EXPORT_PLUGIN(p) {}

} // namespace akari