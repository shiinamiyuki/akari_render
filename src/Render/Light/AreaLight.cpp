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
#include <Akari/Core/Plugin.h>
#include <Akari/Core/Spectrum.h>
#include <Akari/Render/Light.h>
#include <Akari/Render/Material.h>
#include <Akari/Render/Mesh.h>

namespace Akari {
    class AreaLight final : public Light {
        const Mesh *mesh = nullptr;
        int primId{};
        Triangle triangle{};
        Float area = 0.0f;
        Emission emission;
        CoordinateSystem localFrame;
      public:
        AreaLight() = default;
        AreaLight(const Mesh *mesh, int primId) : mesh(mesh), primId(primId) {
            mesh->GetTriangle(primId, &triangle);
            area = triangle.Area();
            auto mat = mesh->GetMaterialSlot(mesh->GetPrimitiveGroup(primId));
            emission = mat.emission;
            localFrame = CoordinateSystem(triangle.Ng);
        }
        AKR_DECL_COMP(AreaLight, "AreaLight")
        Spectrum Li(const vec3 &wo, ShadingPoint &sp) const override {
            if (dot(wo, triangle.Ng) < 0) {
                return Spectrum(0);
            }
            if (emission.strength && emission.color) {
                return emission.color->Evaluate(sp) * emission.strength->Evaluate(sp);
            }
            return Spectrum(0);
        }
        void SampleIncidence(const vec2 &u, const Interaction &ref, RayIncidentSample *sample,
                             VisibilityTester *tester) const override {
            SurfaceSample surfaceSample{};
            triangle.Sample(u, &surfaceSample);
            auto wi = surfaceSample.p - ref.p;
            auto dist2 = dot(wi, wi);
            auto dist = std::sqrt(dist2);
            wi /= dist;

            ShadingPoint sp{};
            sp.texCoords = triangle.InterpolatedNormal(surfaceSample.uv);
            sample->I = Li(-wi, sp);
            sample->wi = wi;
            sample->pdf = dist2 / (-dot(sample->wi, surfaceSample.normal)) * surfaceSample.pdf;
            sample->normal = surfaceSample.normal;

            tester->shadowRay =
                Ray(surfaceSample.p, -1.0f * wi, Eps / abs(dot(sample->wi, surfaceSample.normal)), dist * 0.99);
        }
        Float PdfIncidence(const Interaction &ref, const vec3 &wi) const override {
            Intersection _isct;
            Ray ray(ref.p, wi, Eps);
            if (!triangle.Intersect(ray, &_isct)) {
                return 0.0f;
            }
            Float SA = area * (-dot(wi, _isct.Ng)) / (_isct.t * _isct.t);
            return 1.0f / SA;
        }
        Float Power() const override {
            if (emission.strength && emission.color) {
                return area * emission.strength->AverageLuminance() * emission.color->AverageLuminance();
            }
            return 0.0f;
        }
        void PdfEmission(const Ray &ray, Float *pdfPos, Float *pdfDir) const override {
            *pdfPos = 1 / area;
            *pdfDir = std::fmax(0.0f, CosineHemispherePDF(dot(triangle.Ng, ray.d)));
        }
        void SampleEmission(const vec2 &u1, const vec2 &u2, RayEmissionSample *sample) const override {
            SurfaceSample surfaceSample{};
            triangle.Sample(u1, &surfaceSample);
            auto wi = CosineHemisphereSampling(u2);

            sample->pdfPos = surfaceSample.pdf;
            sample->pdfDir = CosineHemispherePDF(wi.y);
            sample->ray = Ray(surfaceSample.p, localFrame.LocalToWorld(wi), Eps);
            ShadingPoint sp{};
            sp.texCoords = triangle.InterpolatedTexCoord(surfaceSample.uv);
            sample->E = Li(-wi, sp);
        }
    };
    AKR_EXPORT_COMP(AreaLight, "Light");

    AKR_EXPORT std::shared_ptr<Light> CreateAreaLight(const Mesh &mesh, int primId) {
        return std::make_shared<AreaLight>(&mesh, primId);
    }
} // namespace Akari