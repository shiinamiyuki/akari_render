

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
#include <akari/core/variant.h>
#include <akari/core/math.h>
#include <akari/render/scenegraph.h>
#include <akari/render/sampling.h>
#include <akari/core/memory.h>
#include <optional>
namespace akari::render {
    struct CameraSample {
        vec2 p_lens;
        vec2 p_film;
        Float weight = 0.0f;
        Vec3 normal;
        Ray ray;
    };
    class PerspectiveCamera {
      public:
        Transform c2w, w2c, r2c, c2r;
        ivec2 _resolution;
        Float fov;
        Float lens_radius = 0.0f;
        Float focal_distance = 0.0f;
        AKR_XPU void preprocess() {
            Transform m;
            m = Transform::scale(Vec3(1.0f / _resolution.x, 1.0f / _resolution.y, 1)) * m;
            m = Transform::scale(Vec3(2, 2, 1)) * m;
            m = Transform::translate(Vec3(-1, -1, 0)) * m;
            m = Transform::scale(Vec3(1, -1, 1)) * m;
            auto s = atan(fov / 2);
            if (_resolution.x > _resolution.y) {
                m = Transform::scale(Vec3(s, s * Float(_resolution.y) / _resolution.x, 1)) * m;
            } else {
                m = Transform::scale(Vec3(s * Float(_resolution.x) / _resolution.y, s, 1)) * m;
            }
            r2c = m;
            c2r = r2c.inverse();
        }

      public:
        PerspectiveCamera(const ivec2 &_resolution, const Transform &c2w, Float fov)
            : c2w(c2w), w2c(c2w.inverse()), _resolution(_resolution), fov(fov) {
            preprocess();
        }
        AKR_XPU ivec2 resolution() const { return _resolution; }
        AKR_XPU CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
            CameraSample sample;
            sample.p_lens = concentric_disk_sampling(u1) * lens_radius;
            sample.p_film = vec2(raster) + u2;
            sample.weight = 1;

            vec2 p = shuffle<0, 1>(r2c.apply_point(Vec3(sample.p_film.x, sample.p_film.y, 0.0f)));
            Ray ray(Vec3(0), Vec3(normalize(Vec3(p.x, p.y, 0) - Vec3(0, 0, 1))));
            if (lens_radius > 0 && focal_distance > 0) {
                Float ft = focal_distance / std::abs(ray.d.z);
                Vec3 pFocus = ray(ft);
                ray.o = Vec3(sample.p_lens.x, sample.p_lens.y, 0);
                ray.d = Vec3(normalize(pFocus - ray.o));
            }
            ray.o = c2w.apply_point(ray.o);
            ray.d = c2w.apply_vector(ray.d);
            sample.normal = c2w.apply_normal(Vec3(0, 0, -1.0f));
            sample.ray = ray;

            return sample;
        }
    };
    class Camera : Variant<const PerspectiveCamera *> {
      public:
        using Variant::Variant;
        AKR_XPU CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
            AKR_VAR_PTR_DISPATCH(generate_ray, u1, u2, raster);
        }
        AKR_XPU ivec2 resolution() const { AKR_VAR_PTR_DISPATCH(resolution); }
    };
    class CameraNode : public SceneGraphNode {
      public:
        virtual Camera *create_camera(Allocator<> *allocator) = 0;
    };
    AKR_EXPORT std::shared_ptr<CameraNode> create_perspective_camera();
} // namespace akari::render
