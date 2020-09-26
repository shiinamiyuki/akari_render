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
#include <akari/common/variant.h>
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
        AKR_XPU void preprocess() {
            Transform3f m;
            m = Transform3f::scale(Vector3f(1.0f / _resolution.x(), 1.0f / _resolution.y(), 1)) * m;
            m = Transform3f::scale(Vector3f(2, 2, 1)) * m;
            m = Transform3f::translate(Point3f(-1, -1, 0)) * m;
            m = Transform3f::scale(Vector3f(1, -1, 1)) * m;
            auto s = atan(fov / 2);
            if (_resolution.x() > _resolution.y()) {
                m = Transform3f::scale(Vector3f(s, s * Float(_resolution.y()) / _resolution.x(), 1)) * m;
            } else {
                m = Transform3f::scale(Vector3f(s * Float(_resolution.x()) / _resolution.y(), s, 1)) * m;
            }
            r2c = m;
            c2r = r2c.inverse();
        }

      public:
        AKR_XPU PerspectiveCamera(const Point2i &_resolution, const Transform3f &c2w, Float fov)
            : c2w(c2w), w2c(c2w.inverse()), _resolution(_resolution), fov(fov) {
            preprocess();
        }
        AKR_XPU Point2i resolution() const { return _resolution; }
        AKR_XPU CameraSample<C> generate_ray(const Point2f &u1, const Point2f &u2, const Point2i &raster) const {
            CameraSample<C> sample;
            sample.p_lens = sampling<C>::concentric_disk_sampling(u1) * lens_radius;
            sample.p_film = Point2f(raster) + u2;
            sample.weight = 1;

            Point2f p = shuffle<0, 1>(r2c.apply_point(Point3f(sample.p_film.x(), sample.p_film.y(), 0.0f)));
            Ray3f ray(Point3f(0), Vector3f(normalize(Point3f(p.x(), p.y(), 0) - Point3f(0, 0, 1))));
            if (lens_radius > 0 && focal_distance > 0) {
                Float ft = focal_distance / std::abs(ray.d.z());
                Point3f pFocus = ray(ft);
                ray.o = Point3f(sample.p_lens.x(), sample.p_lens.y(), 0);
                ray.d = Vector3f(normalize(pFocus - ray.o));
            }
            ray.o = c2w.apply_point(ray.o);
            ray.d = c2w.apply_vector(ray.d);
            sample.normal = c2w.apply_normal(Normal3f(0, 0, -1.0f));
            sample.ray = ray;
            return sample;
        }
    };
    AKR_VARIANT class Camera : public Variant<PerspectiveCamera<C>> {
      public:
        AKR_IMPORT_TYPES()
        using Variant<PerspectiveCamera<C>>::Variant;
        AKR_XPU CameraSample<C> generate_ray(const Point2f &u1, const Point2f &u2, const Point2i &raster) const {
            AKR_VAR_DISPATCH(generate_ray, u1, u2, raster);
        }
        AKR_XPU Point2i resolution() const { AKR_VAR_DISPATCH(resolution); }
    };
} // namespace akari