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
#include <Akari/Render/Accelerator.h>
#include <embree3/rtcore.h>

namespace Akari {
    class EmbreeAccelerator : public Accelerator {
      public:
        AKR_DECL_COMP(EmbreeAccelerator, "EmbreeAccelerator")
        void Build(const Scene &scene) override {}
        bool Intersect(const Ray &ray, Intersection *intersection) const override { return false; }
        bool Occlude(const Ray &ray) const override { return false; }
        Bounds3f GetBounds() const override { return Akari::Bounds3f(); }
    };
    AKR_EXPORT_COMP(EmbreeAccelerator, "Accelerator")
} // namespace Akari