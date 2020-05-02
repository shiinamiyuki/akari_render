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

#ifndef AKARIRENDER_ACCELERATOR_H
#define AKARIRENDER_ACCELERATOR_H
#include <Akari/Core/Component.h>
#include <Akari/Render/Geometry.hpp>
namespace Akari {
    class Scene;
    class AKR_EXPORT Accelerator : public Component {
      public:
        virtual void buil(const Scene &scene) = 0;
        virtual bool intersect(const Ray &ray, Intersection *intersection) const = 0;
        [[nodiscard]] virtual bool occlude(const Ray &ray) const = 0;
        virtual Bounds3f get_bounds()const =0 ;
    };
} // namespace Akari
#endif // AKARIRENDER_ACCELERATOR_H
