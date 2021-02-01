// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "common.h"

namespace akari::gpu::kernel {
    struct CameraSample {
        vec2 p_lens;
        vec2 p_film;
        Float weight = 0.0f;
        Vec3 normal;
        Ray ray;
    };
    struct PerspectiveCamera {
        Transform c2w, w2c, r2c, c2r;
        ivec2 _resolution;
        Float fov;
        Float lens_radius    = 0.0f;
        Float focal_distance = 0.0f;
        AKR_XPU CameraSample generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster) const {
            CameraSample sample;
            using namespace akari::render;
            sample.p_lens = concentric_disk_sampling(u1) * lens_radius;
            sample.p_film = vec2(raster) + u2;
            sample.weight = 1;

            vec2 p = shuffle<0, 1>(r2c.apply_point(Vec3(sample.p_film.x, sample.p_film.y, 0.0f)));
            Ray ray(Vec3(0), Vec3(normalize(Vec3(p.x, p.y, 0) - Vec3(0, 0, 1))));
            if (lens_radius > 0 && focal_distance > 0) {
                Float ft    = focal_distance / std::abs(ray.d.z);
                Vec3 pFocus = ray(ft);
                ray.o       = Vec3(sample.p_lens.x, sample.p_lens.y, 0);
                ray.d       = Vec3(normalize(pFocus - ray.o));
            }
            ray.o         = c2w.apply_point(ray.o);
            ray.d         = c2w.apply_vector(ray.d);
            sample.normal = c2w.apply_normal(Vec3(0, 0, -1.0f));
            sample.ray    = ray;

            return sample;
        }
    };
    struct CameraHandle {
        union {
            PerspectiveCamera *perspective;
        };
        enum class Type { Perspective };
        Type type;
    };
} // namespace akari::gpu::kernel