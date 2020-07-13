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

#include <random>
#include <akari/core/math.h>
#include <akari/core/plugin.h>
#include <akari/core/spectrum.h>
#include <akari/render/light.h>
#include <akari/render/material.h>
#include <akari/render/mesh.h>
#include <akari/core/lazy.hpp>
#include <akari/core/parallel.h>


namespace akari {
    class AreaLight final : public Light {
        const Mesh *mesh = nullptr;
        int primId{};
        Triangle triangle{};
        Float area = 0.0f;
        Emission emission;
        Frame3f localFrame;
        Float _power = 0.0f;
      public:
        LightType get_light_type() const override { return LightType::ENone; }
        AreaLight() = default;
        AreaLight(const Mesh *mesh, int primId) : mesh(mesh), primId(primId) {
            mesh->get_triangle(primId, &triangle);
            area = triangle.Area();
            auto mat = mesh->get_material_slot(mesh->get_primitive_group(primId));
            emission = mat.emission;
            localFrame = Frame3f(triangle.Ng);
        }
        AKR_IMPLS(Light)
        void commit()override {
            Component::commit();
            if(!emission.color || !emission.strength)
                return;
            auto v0 = vec3(triangle.texCoords[0], 0);
            auto v1 = vec3(triangle.texCoords[1], 1);
            auto v2 = vec3(triangle.texCoords[2], 2);
            auto _tex_area = std::min(1.0f, std::abs(length(cross(v1 - v0, v2 - v0))));
            auto hint1 = emission.color->importance_map_resolution_hint();
            auto hint2 = emission.strength->importance_map_resolution_hint();
            size_t total_samples = std::max(hint1.x, hint2.x) * std::max(hint1.y, hint2.y);
            size_t samples = std::max<size_t>(1, _tex_area * total_samples);
            std::random_device rd;
            std::uniform_real_distribution<Float> dist;

            _power = 0;
            for(size_t i =0 ;i <samples;i++){
                vec2 uv(dist(rd),dist(rd));
                auto tex_coord = triangle.interpolated_tex_coord(uv);
                ShadingPoint sp{tex_coord};
                _power += (emission.color->evaluate(sp) * emission.strength->evaluate(sp)).luminance();
            }
            
        }
        Spectrum Li(const vec3 &wo, const vec2 &uv) const override {
            ShadingPoint sp;
            sp.texCoords = triangle.interpolated_tex_coord(uv);
            if (dot(wo, triangle.Ng) < 0) {
                return Spectrum(0);
            }
            if (emission.strength && emission.color) {
                return emission.color->evaluate(sp) * emission.strength->evaluate(sp);
            }
            return Spectrum(0);
        }
        void sample_incidence(const vec2 &u, const Interaction &ref, RayIncidentSample *sample,
                              VisibilityTester *tester) const override {
            SurfaceSample surfaceSample{};
            triangle.sample_surface(u, &surfaceSample);
            auto wi = surfaceSample.p - ref.p;
            auto dist2 = dot(wi, wi);
            auto dist = std::sqrt(dist2);
            wi /= dist;

            sample->I = Li(-wi, surfaceSample.uv);
            sample->wi = wi;
            sample->pdf = dist2 / (-dot(sample->wi, surfaceSample.normal)) * surfaceSample.pdf;
            sample->normal = surfaceSample.normal;

            tester->shadowRay =
                Ray(surfaceSample.p, -1.0f * wi, Eps() / abs(dot(sample->wi, surfaceSample.normal)), dist * 0.99);
        }
        Float pdf_incidence(const Interaction &ref, const vec3 &wi) const override {
            Intersection _isct;
            Ray ray(ref.p, wi);
            if (!triangle.Intersect(ray, &_isct)) {
                return 0.0f;
            }
            Float SA = area * (-dot(wi, _isct.Ng)) / (_isct.t * _isct.t);
            return 1.0f / SA;
        }
        Float power() const override {
            return _power;
        }
        void pdf_emission(const Ray &ray, Float *pdfPos, Float *pdfDir) const override {
            *pdfPos = 1 / area;
            *pdfDir = std::fmax(0.0f, cosine_hemisphere_pdf(dot(triangle.Ng, ray.d)));
        }
        void sample_emission(const vec2 &u1, const vec2 &u2, RayEmissionSample *sample) const override {
            SurfaceSample surfaceSample{};
            triangle.sample_surface(u1, &surfaceSample);
            auto wi = cosine_hemisphere_sampling(u2);

            sample->pdfPos = surfaceSample.pdf;
            sample->pdfDir = cosine_hemisphere_pdf(wi.y);
            wi = localFrame.local_to_world(wi);
            sample->ray = Ray(surfaceSample.p, wi, Eps());
            sample->normal = surfaceSample.normal;
            ShadingPoint sp{};
            sp.texCoords = triangle.interpolated_tex_coord(surfaceSample.uv);
            sample->uv = surfaceSample.uv;
            sample->E = Li(wi, sample->uv);
        }
    };
#include "generated/AreaLight.hpp"
    AKR_EXPORT_PLUGIN(p){

    }

    AKR_EXPORT std::shared_ptr<Light> CreateAreaLight(const Mesh &mesh, int primId) {
        return std::make_shared<AreaLight>(&mesh, primId);
    }
} // namespace akari