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

      public:
        AreaLight() = default;
        AreaLight(const Mesh *mesh, int primId) : mesh(mesh), primId(primId) {
            mesh->GetTriangle(primId, &triangle);
            area = triangle.Area();
            auto mat = mesh->GetMaterialSlot(mesh->GetPrimitiveGroup(primId));
            emission = mat.emission;
        }
        AKR_DECL_COMP(AreaLight, "AreaLight")
        Spectrum Li(ShadingPoint &sp) const override {
            if (emission.strength && emission.color) {
                return emission.color->Evaluate(sp) * emission.strength->Evaluate(sp);
            }
            return Spectrum(0);
        }
        void SampleLi(const vec2 &u, Intersection &isct, LightSample &sample, VisibilityTester &tester) const override {
            SurfaceSample surfaceSample{};
            triangle.Sample(u, &surfaceSample);
            auto wi = surfaceSample.p - isct.p;
            auto dist2 = dot(wi, wi);
            auto dist = std::sqrt(dist2);
            wi /= dist;

            ShadingPoint sp{};
            sp.texCoords = triangle.InterpolatedNormal(surfaceSample.uv);
            sample.Li = Li(sp);
            sample.wi = wi;
            sample.pdf = dist2 / (-dot(sample.wi, surfaceSample.normal)) * surfaceSample.pdf;
            sample.normal = surfaceSample.normal;

            tester.shadowRay =
                Ray(surfaceSample.p, -1.0f * wi, Eps / abs(dot(sample.wi, surfaceSample.normal)), dist * 0.99);
        }
        Float PdfLi(const Intersection &intersection, const vec3 &wi) const override {
            Intersection _isct;
            Ray ray(intersection.p, wi, Eps);
            if (!triangle.Intersect(ray, &_isct)) {
                return 0.0f;
            }
            Float SA = triangle.Area() * (-dot(wi, _isct.Ng)) / (_isct.t * _isct.t);
            return 1.0f / SA;
        }
        Float Power() const override {
            if (emission.strength && emission.color) {
                return triangle.Area() * emission.strength->AverageLuminance() * emission.color->AverageLuminance();
            }
            return 0.0f;
        }
    };
    AKR_EXPORT_COMP(AreaLight, "Light");

    AKR_EXPORT std::shared_ptr<Light> CreateAreaLight(const Mesh &mesh, int primId) {
        return std::make_shared<AreaLight>(&mesh, primId);
    }
} // namespace Akari