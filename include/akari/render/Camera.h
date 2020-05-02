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

#ifndef AKARIRENDER_CAMERA_H
#define AKARIRENDER_CAMERA_H

#include "Light.h"
#include <akari/Core/Component.h>
#include <akari/Core/Film.h>
#include <akari/Render/Geometry.hpp>
#include <akari/Render/EndPoint.h>

namespace akari {
    struct CameraSample {
        vec2 p_film;
        vec2 p_lens;
        Ray primary;
        Float weight = 1;
        vec3 normal;
    };


    class Camera : public EndPoint {
      public:
        [[nodiscard]] virtual bool IsProjective() const { return false; }

        virtual void generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster, CameraSample *sample) const = 0;

        [[nodiscard]] virtual std::shared_ptr<Film> GetFilm() const = 0;

        virtual Spectrum We(const Ray &ray, vec2 *pRaster) const {
            AKARI_PANIC("Camera::We(const Ray &, vec2 &) is not implemented");
        }
        void pdf_emission(const Ray &ray, Float *pdfPos, Float *pdfDir) const override{
            AKARI_PANIC("Camera::PdfWe(const Ray &ray, Float *pdfPos, Float *pdfDir) is not implemented");
        }
        Float pdf_incidence(const Interaction& ref, const vec3& wi) const override{
            return 0;
        }
        void sample_emission(const vec2& u1, const vec2& u2, RayEmissionSample* sample) const override{
            AKARI_PANIC("void Camera::SampleEmission(const vec2& u1, const vec2& u2, RayEmissionSample* sample)  is not implemented");
        }

    };

} // namespace akari
#endif // AKARIRENDER_CAMERA_H
