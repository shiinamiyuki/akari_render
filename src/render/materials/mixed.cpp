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

#include <utility>
#include <akari/core/plugin.h>
#include <akari/render/geometry.hpp>
#include <akari/render/material.h>
#include <akari/render/reflection.h>
#include <akari/render/texture.h>
#include <akari/plugins/mixed_material.h>

namespace akari {
    class MixedMaterial : public Material {
        [[refl]] std::shared_ptr<Texture> fraction;
        [[refl]] std::shared_ptr<Material> matA;
        [[refl]] std::shared_ptr<Material> matB;
        

      public:
        MixedMaterial() = default;
        MixedMaterial(const std::shared_ptr<Texture> &fraction, const std::shared_ptr<Material> &matA,
                      const std::shared_ptr<Material> &matB)
            : fraction(fraction), matA(matA), matB(matB) {}

        AKR_IMPLS(Material)
        void compute_scattering_functions(SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,
                                          Float scale) const override {
            auto f = fraction->evaluate(si->sp)[0];
            matA->compute_scattering_functions(si, arena, mode, scale * f);
            matB->compute_scattering_functions(si, arena, mode, scale * (1.0f - f));
        }
        [[refl]] bool support_bidirectional() const override {
            return matA->support_bidirectional() && matB->support_bidirectional();
        }
    };
    std::shared_ptr<Material> create_mixed_material(const std::shared_ptr<Texture> &fraction,
                                                    const std::shared_ptr<Material> &matA,
                                                    const std::shared_ptr<Material> &matB) {
        return std::make_shared<MixedMaterial>(fraction, matA, matB);
    }
#include "generated/MixedMaterial.hpp"
    AKR_EXPORT_PLUGIN(p) {}
} // namespace akari