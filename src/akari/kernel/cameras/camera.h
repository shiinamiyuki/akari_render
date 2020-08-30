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
#include <akari/common/taggedpointer.h>
#include <akari/common/math.h>
namespace akari {
    AKR_VARIANT struct CameraSample {
        using Float = typename C::Float;
        AKR_IMPORT_CORE_TYPES()
        Point2f p_lens;
        Point2f p_film;
        Float weight = 0.0f;
        Normal3f normal;
        Ray<C> ray;
    };
    AKR_VARIANT class PerspectiveCamera {
      public:
        AKR_IMPORT_TYPES()
      private:
        Transform3f c2w, w2c, r2c, c2r;
        Point2i _resolution;
        Float fov;
        Float lens_radius = 0.0f;
        Float focal_distance = 0.0f;
        void preprocess();

      public:
        PerspectiveCamera(const Point2i &_resolution, const Transform3f &c2w, Float fov)
            : c2w(c2w), w2c(c2w.inverse()), _resolution(_resolution), fov(fov) {
            preprocess();
        }
        Point2i resolution() const { return _resolution; }
        void generate_ray(const Point2f &u1, const Point2f &u2, const Point2i &raster, CameraSample<C> *sample) const;
    };
    AKR_VARIANT class Camera : public TaggedPointer<PerspectiveCamera<C>> {
      public:
        AKR_IMPORT_TYPES()
        using TaggedPointer<PerspectiveCamera<C>>::TaggedPointer;
        void generate_ray(const Point2f &u1, const Point2f &u2, const Point2i &raster, CameraSample<C> *sample) const {
            AKR_TAGGED_DISPATCH(generate_ray, u1, u2, raster, sample);
        }
        Point2i resolution() const { AKR_TAGGED_DISPATCH(resolution); }
    };
} // namespace akari