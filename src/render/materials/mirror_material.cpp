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
#include <akari/render/texture.h>
#include <utility>

namespace akari {
    class MirrorMaterial final : public Material {
        std::shared_ptr<Texture> color;

      public:
        MirrorMaterial() = default;
        explicit MirrorMaterial(std::shared_ptr<Texture> color) : color(std::move(color)) {}
        AKR_SER(color)
        AKR_DECL_COMP(MirrorMaterial, "MirrorMaterial")
        void compute_scattering_functions(SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,
                                          Float scale) const override {
            si->bsdf = arena.alloc<BSDF>(*si);
            auto c = color->evaluate(si->sp);
            si->bsdf->add_component(arena.alloc<SpecularReflection>(c * scale, arena.alloc<FresnelNoOp>()));
        }
        void commit() override { color->commit(); }
        bool support_bidirectional() const override { return true; }
    };
    AKR_PLUGIN_ON_LOAD {
        // clang-format off
        class_<MirrorMaterial,Material>("MirrorMaterial")
            .property("color",&MirrorMaterial::color);
        //clang-format on
    }
    AKR_EXPORT_COMP(MirrorMaterial, "Material")

} // namespace akari