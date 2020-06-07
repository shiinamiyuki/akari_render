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

#ifndef AKARIRENDER_MATERIAL_H
#define AKARIRENDER_MATERIAL_H

#include <akari/core/component.h>
#include <akari/core/memory_arena.hpp>
#include <akari/render/bsdf.h>
#include <akari/render/interaction.h>
#include <akari/render/texture.h>
#include <akari/core/serialize.hpp>
#include <akari/core/detail/serialize-impl.hpp>
namespace akari {
    class Material;
    class Light;
    struct ScatteringEvent;

    class AKR_EXPORT Material : public Component {
      public:
        virtual void compute_scattering_functions(SurfaceInteraction *si, MemoryArena &arena, TransportMode mode,
                                                  Float scale) const = 0;

        // Materials like Toon-ish shading would not satisfy this property
        virtual bool support_bidirectional() const = 0;
    };

    struct Emission {
        std::shared_ptr<Texture> color;
        std::shared_ptr<Texture> strength;
        AKR_SER(color, strength)
    };

    struct MaterialSlot {
        std::string name;
        std::shared_ptr<Material> material;
        Emission emission;
        bool marked_as_light = false;
        AKR_SER(material, name, marked_as_light, emission)
    };
} // namespace akari
#endif // AKARIRENDER_MATERIAL_H
