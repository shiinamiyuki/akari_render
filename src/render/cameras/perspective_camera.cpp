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
#include <akari/render/camera.h>
#include <akari/render/sampling.hpp>
namespace akari {
    class PerspectiveCamera final : public Camera {
        std::shared_ptr<Film> film;
        ivec2 resolution = ivec2(500, 500);
        float lens_radius = 0;
        Angle<float> fov = {DegreesToRadians(80.0f)};
        Transform cameraToWorld, worldToCamera;
        AffineTransform transform;
        Float focal_distance = 1;
        Transform rasterToCamera{}, cameraToRaster{};

      public:
        AKR_DECL_COMP()

        AKR_SER(resolution, lens_radius, fov, transform, focal_distance)
        PerspectiveCamera() : cameraToWorld(identity<mat4>()), worldToCamera(identity<mat4>()) {}
        [[nodiscard]] bool IsProjective() const override { return true; }

        void generate_ray(const vec2 &u1, const vec2 &u2, const ivec2 &raster, CameraSample *sample) const override {
            sample->p_lens = concentric_disk_sampling(u1) * lens_radius;
            sample->p_film = vec2(raster) + (u2 - 0.5f);
            sample->weight = 1;

            vec2 p = vec2(rasterToCamera.ApplyPoint(Vector3f(sample->p_film, 0)));
            Ray ray(Vector3f(0), normalize(Vector3f(p, 0) - Vector3f(0, 0, 1)));
            if (lens_radius > 0 && focal_distance > 0) {
                Float ft = focal_distance / abs(ray.d.z);
                Vector3f pFocus = ray.At(ft);
                ray.o = Vector3f(sample->p_lens, 0);
                ray.d = normalize(pFocus - ray.o);
            }
            ray.o = cameraToWorld.ApplyPoint(ray.o);
            ray.d = cameraToWorld.ApplyNormal(ray.d);
            sample->normal = cameraToWorld.ApplyNormal(Vector3f(0, 0, -1));
            ;
            sample->primary = ray;
        }

        void commit() override {
            film = std::make_shared<Film>(resolution);
            cameraToWorld = Transform(transform.ToMatrix4());
            worldToCamera = cameraToWorld.Inverse();
            mat4 m = identity<mat4>();
            m = scale(mat4(1.0), Vector3f(1.0f / resolution.x, 1.0f / resolution.y, 1)) * m;
            m = scale(mat4(1.0), Vector3f(2, 2, 1)) * m;
            m = translate(mat4(1.0), Vector3f(-1, -1, 0)) * m;
            m = scale(mat4(1.0), Vector3f(1, -1, 1)) * m;
            auto s = std::atan(fov.value / 2);
            if (resolution.x > resolution.y) {
                m = scale(mat4(1.0), Vector3f(s, s * float(resolution.y) / resolution.x, 1)) * m;
            } else {
                m = scale(mat4(1.0), Vector3f(s * float(resolution.x) / resolution.y, s, 1)) * m;
            }
            rasterToCamera = Transform(m);
            cameraToRaster = rasterToCamera.Inverse();
        }
        [[nodiscard]] std::shared_ptr<Film> GetFilm() const override { return film; }
        Spectrum We(const Ray &ray, vec2 *pRaster) const override {
            Float cosTheta = dot(cameraToWorld.ApplyVector(Vector3f(0, 0, -1)), ray.d);
            Vector3f pFocus = ray.At((lens_radius == 0 ? 1 : focal_distance) / cosTheta);
            vec2 raster = cameraToRaster.ApplyPoint(worldToCamera.ApplyPoint(pFocus));
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
            Vector3f pMin = Vector3f(0);
            Vector3f pMax = Vector3f(film->Dimension(), 0);
            pMin = rasterToCamera.ApplyPoint(pMin);
            pMax = rasterToCamera.ApplyPoint(pMax);
            return std::abs((pMax.y - pMin.y) * (pMax.x - pMin.x));
        }
        void pdf_emission(const Ray &ray, Float *pdfPos, Float *pdfDir) const override {
            Float cosTheta = dot(cameraToWorld.ApplyVector(Vector3f(0, 0, -1)), ray.d);
            Vector3f pFocus = ray.At((lens_radius == 0 ? 1 : focal_distance) / cosTheta);
            vec2 raster = cameraToRaster.ApplyPoint(worldToCamera.ApplyPoint(pFocus));
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

            Vector3f pLensWorld = cameraToWorld.ApplyPoint(Vector3f(pLens, 0));
            sample->normal = cameraToWorld.ApplyNormal(Vector3f(0, 0, -1));
            sample->wi = pLensWorld - ref.p;
            Float dist = length(sample->wi);
            sample->wi /= dist;
            Float lensArea = lens_radius == 0 ? 1.0f : lens_radius * lens_radius * Pi;
            tester->shadowRay = Ray(pLensWorld, -sample->wi, Eps(), dist * (1.0 - ShadowEps()));
            sample->pdf = (dist * dist) / (lensArea * abs(dot(sample->normal, sample->wi)));
            sample->I = We(tester->shadowRay, &sample->pos);
        }
    };
    AKR_EXPORT_PLUGIN(PerspectiveCamera, p) {
        auto c = class_<PerspectiveCamera, Camera, Component>("PerspectiveCamera");
        c.constructor<>();
        c.method("save", &PerspectiveCamera::save);
        c.method("load", &PerspectiveCamera::load);
        c.property("fov", &PerspectiveCamera::fov);
    }

} // namespace akari