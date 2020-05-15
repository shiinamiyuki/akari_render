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

#include <akari/core/config.h>
#include <akari/core/logger.h>
#include <akari/core/math.h>
#include <akari/core/plugin.h>
#include <akari/core/sampling.hpp>
#include <akari/render/camera.h>
namespace akari {
    class PerspectiveCamera final : public Camera {
        std::shared_ptr<Film> film;
        [[refl]]ivec2 resolution = ivec2(500, 500);
        [[refl]]float lens_radius = 0;
        [[refl]]Angle<float> fov = {DegreesToRadians(80.0f)};
        Transform cameraToWorld, worldToCamera;
        [[refl]] AffineTransform transform;
        [[refl]] Float focal_distance = 1;
        Transform rasterToCamera{}, cameraToRaster{};

      public:
        AKR_IMPLS(Camera, EndPoint)
        PerspectiveCamera() : cameraToWorld(identity<mat4>()), worldToCamera(identity<mat4>()) {}
        [[nodiscard]] bool IsProjective() const override { return true; }

        void generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster, CameraSample *sample) const override {
            sample->p_lens = concentric_disk_sampling(u1) * lens_radius;
            sample->p_film = vec2(raster) + (u2 - 0.5f);
            sample->weight = 1;

            vec2 p = vec2(rasterToCamera.apply_point(vec3(sample->p_film, 0)));
            Ray ray(vec3(0), normalize(vec3(p, 0) - vec3(0, 0, 1)));
            if (lens_radius > 0 && focal_distance > 0) {
                Float ft = focal_distance / abs(ray.d.z);
                vec3 pFocus = ray.At(ft);
                ray.o = vec3(sample->p_lens, 0);
                ray.d = normalize(pFocus - ray.o);
            }
            ray.o = cameraToWorld.apply_point(ray.o);
            ray.d = cameraToWorld.apply_normal(ray.d);
            sample->normal = cameraToWorld.apply_normal(vec3(0, 0, -1));
            ;
            sample->primary = ray;
        }

        void commit() override {
            film = std::make_shared<Film>(resolution);
            cameraToWorld = Transform(transform.ToMatrix4());
            worldToCamera = cameraToWorld.inverse();
            mat4 m = identity<mat4>();
            m = scale(mat4(1.0), vec3(1.0f / resolution.x, 1.0f / resolution.y, 1)) * m;
            m = scale(mat4(1.0), vec3(2, 2, 1)) * m;
            m = translate(mat4(1.0), vec3(-1, -1, 0)) * m;
            m = scale(mat4(1.0), vec3(1, -1, 1)) * m;
            auto s = std::atan(fov.value / 2);
            if (resolution.x > resolution.y) {
                m = scale(mat4(1.0), vec3(s, s * float(resolution.y) / resolution.x, 1)) * m;
            } else {
                m = scale(mat4(1.0), vec3(s * float(resolution.x) / resolution.y, s, 1)) * m;
            }
            rasterToCamera = Transform(m);
            cameraToRaster = rasterToCamera.inverse();
        }
        [[nodiscard]] std::shared_ptr<Film> GetFilm() const override { return film; }
        Spectrum We(const Ray &ray, vec2 *pRaster) const override {
            Float cosTheta = dot(cameraToWorld.apply_vector(vec3(0, 0, -1)), ray.d);
            vec3 pFocus = ray.At((lens_radius == 0 ? 1 : focal_distance) / cosTheta);
            vec2 raster = cameraToRaster.apply_point(worldToCamera.apply_point(pFocus));
            (void)raster;
            if (cosTheta <= 0) {
                return Spectrum(0);
            }
            auto bounds = film->bounds();
            if (raster.x < bounds.p_min.x || raster.x > bounds.p_max.x || raster.y < bounds.p_min.y ||
                raster.y > bounds.p_max.y) {
                return Spectrum(0);
            }
            //            Info("raster {}, {}, A(): {}\n", raster.x,raster.y, A());
            *pRaster = raster;
            Float lensArea = lens_radius == 0 ? 1.0f : lens_radius * lens_radius * Pi;
            return Spectrum(1 / (A() * lensArea * Power<4>(cosTheta)));
        }
        Float A() const {
            vec3 pMin = vec3(0);
            vec3 pMax = vec3(film->resolution(), 0);
            pMin = rasterToCamera.apply_point(pMin);
            pMax = rasterToCamera.apply_point(pMax);
            return std::abs((pMax.y - pMin.y) * (pMax.x - pMin.x));
        }
        void pdf_emission(const Ray &ray, Float *pdfPos, Float *pdfDir) const override {
            Float cosTheta = dot(cameraToWorld.apply_vector(vec3(0, 0, -1)), ray.d);
            vec3 pFocus = ray.At((lens_radius == 0 ? 1 : focal_distance) / cosTheta);
            vec2 raster = cameraToRaster.apply_point(worldToCamera.apply_point(pFocus));
            (void)raster;
            if (cosTheta <= 0) {
                return;
            }

            Float d = 1 / cosTheta;
            // pw = pa(p) * da/dw = 1 / filmArea * d^2 / cosTheta
            Float lensArea = lens_radius == 0 ? 1.0f : lens_radius * lens_radius * Pi;
            *pdfPos = 1 / lensArea;
            *pdfDir = 1 / A() * d * d / cosTheta;
        }

        void sample_incidence(const vec2 &u, const Interaction &ref, RayIncidentSample *sample,
                              VisibilityTester *tester) const override {
            vec2 pLens = lens_radius * concentric_disk_sampling(u);

            vec3 pLensWorld = cameraToWorld.apply_point(vec3(pLens, 0));
            sample->normal = cameraToWorld.apply_normal(vec3(0, 0, -1));
            sample->wi = pLensWorld - ref.p;
            Float dist = length(sample->wi);
            sample->wi /= dist;
            Float lensArea = lens_radius == 0 ? 1.0f : lens_radius * lens_radius * Pi;
            tester->shadowRay = Ray(pLensWorld, -sample->wi, Eps(), dist * (1.0 - ShadowEps()));
            sample->pdf = (dist * dist) / (lensArea * abs(dot(sample->normal, sample->wi)));
            sample->I = We(tester->shadowRay, &sample->pos);
        }
    };
#include "generated/PerspectiveCamera.hpp"
    AKR_EXPORT_PLUGIN(p){}

} // namespace akari