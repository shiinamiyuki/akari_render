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
#include <akari/core/distribution.h>
#include <akari/core/image.h>
#include <akari/core/film.h>
#include <akari/render/scene.h>
#include <akari/render/scenegraph.h>
#include <akari/render/integrator.h>
namespace akari::render {

    class Denoiser {
      public:
        virtual void add_aov_requests(RenderInput &inputs) = 0;
        virtual std::optional<Image> denoise(const Scene *scene, RenderOutput &aov) = 0;
        virtual ~Denoiser() = default;
    };
    
    AKR_EXPORT Image nlmeans(const Image &image, const Image &guide, const Image &variance, uint32_t F, uint32_t R,
                             double k);
} // namespace akari::render