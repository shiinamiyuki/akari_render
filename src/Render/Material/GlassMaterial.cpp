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

#include <Akari/Core/Plugin.h>
#include <Akari/Render/Geometry.hpp>
#include <Akari/Render/Material.h>
#include <Akari/Render/Plugins/Matte.h>
#include <Akari/Render/Reflection.h>
#include <Akari/Render/Texture.h>
#include <utility>

namespace Akari {
    class GlassMaterial final : public Material {
        std::shared_ptr<Texture> color;

      public:
        GlassMaterial() = default;
        explicit GlassMaterial(std::shared_ptr<Texture> color) : color(std::move(color)) {}
        AKR_SER(color)
        AKR_DECL_COMP(GlassMaterial, "GlassMaterial")
        void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,
                                        Float scale) const override {
            auto c = color->Evaluate(si->sp);
            si->bsdf->AddComponent(arena.alloc<SpecularReflection>(c * scale));
        }
    };

    AKR_EXPORT_COMP(GlassMaterial, "Material")

} // namespace Akari