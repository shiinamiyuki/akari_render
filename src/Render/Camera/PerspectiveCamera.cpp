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

#include <Akari/Core/Config.h>
#include <Akari/Core/Math.h>
#include <Akari/Core/Plugin.h>
#include <Akari/Core/Sampling.hpp>
#include <Akari/Render/Camera.h>
namespace Akari {
    class PerspectiveCamera final: public Camera {
        std::shared_ptr<Film> film;
        ivec2 filmDimension = ivec2(500,500);
        float lensRadius = 0;
        Angle<float> fov = {DegreesToRadians(80.0f)};
        Transform _transform, inv_transform;
        AffineTransform transform;

      public:
        AKR_DECL_COMP(PerspectiveCamera, "PerspectiveCamera")

        AKR_SER(filmDimension, lensRadius, fov, transform)
        PerspectiveCamera() : _transform(identity<mat4>()), inv_transform(identity<mat4>()) {}
        [[nodiscard]] bool IsProjective() const override { return true; }

        void GenerateRay(const vec2 &u1, const vec2 &u2, const ivec2 &raster, CameraSample &sample) const override {
            sample.p_lens = ConcentricSampleDisk(u1) * lensRadius;
            sample.p_film = vec2(raster) + (u2 - 0.5f);
            sample.weight = 1;

            vec2 p = sample.p_film / vec2(filmDimension);
            p = 2.0f * p - 1.0f;
            p.y = -p.y;
            if (filmDimension.x > filmDimension.y) {
                p.y *= float(filmDimension.y) / filmDimension.x;
            } else {
                p.x *= float(filmDimension.x) / filmDimension.y;
            }
            auto z = 1 / std::atan(fov.value / 2);
            vec3 ro = _transform.ApplyPoint(vec3(0));
            vec3 rd = _transform.ApplyVector(normalize(vec3(p, 0) - vec3(0, 0, z)));
            sample.primary = Ray(ro, rd, GetConfig()->RayBias);
        }

        void Commit() override {
            film = std::make_shared<Film>(filmDimension);
            _transform = Transform(transform.ToMatrix4());
            inv_transform = _transform.Inverse();
        }
        [[nodiscard]] std::shared_ptr<Film> GetFilm() const override { return film; }
    };
    AKR_EXPORT_COMP(PerspectiveCamera, "Camera")
} // namespace Akari