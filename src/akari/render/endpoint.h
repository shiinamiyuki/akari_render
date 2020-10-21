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
namespace akari::render {
    struct ReferencePoint {
        Vec3 p;
        Vec3 ng;
    };
    template <class RayIncidenceSample, class RayEmissionSample, class IncidenceSampleContext>
    class EndPoint {
      public:
        /*
         struct RayIncidenceSample {
          vec3 wi;
          Spectrum I;
          vec3 ng;
          vec2 pos; // the uv coordinate of the sampled position or the raster position (0,0) to (bound.x, boubd.y)
          float pdf;
      };
      struct RayEmissionSample {
          Ray ray;
          Spectrum E;
          vec3 ng;
          vec2 uv; // 2D parameterized position
          float pdfPos, pdfDir;
      };
        */
        virtual Float pdf_incidence(const ReferencePoint &ref, const vec3 &wi) const {
            AKR_PANIC("pdf_incidence not implemented!");
        }
        virtual std::pair<Float, Float> pdf_emission(const Ray &ray) const {
            AKR_PANIC("pdf_emission not implemented!");
        }
        virtual RayIncidenceSample sample_incidence(const IncidenceSampleContext &ctx) const = 0;
        virtual RayEmissionSample sample_emission(const vec2 &u1, const vec2 &u2) const {
            AKR_PANIC("sample_emission not implemented!");
        }
    };
} // namespace akari::render