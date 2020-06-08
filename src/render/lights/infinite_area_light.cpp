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

#include <akari/render/distribution.hpp>
#include <akari/plugins/infinite_area_light.h>
#include <akari/core/parallel.h>
#include <akari/core/plugin.h>
#include <akari/render/scene.h>

namespace akari {

    class InfiniteAreaLightImpl : public Light {
        std::shared_ptr<Texture> color;
        std::unique_ptr<Distribution2D> distribution;
        TransformManipulator transform;
        Transform world_to_light, light_to_world;
        const size_t distribution_map_resolution = 4096;
        Bounds3f world_bounds;
        Float world_radius;
        Float _power = 0.0f;

      public:
        AKR_IMPLS(Light)
        InfiniteAreaLightImpl() = default;
        InfiniteAreaLightImpl(std::shared_ptr<Texture> color, const Bounds3f &world_bounds)
            : color(color), world_bounds(world_bounds) {
            world_to_light = Transform(transform.matrix4());
            light_to_world = world_to_light.inverse();
            world_radius = length(world_bounds.p_max - world_bounds.centroid());
        }
        Float power() const override { return _power; }
        static vec2 get_uv(const vec3 &wi) {
            auto theta = spherical_theta(wi);
            // auto sinTheta = std::sin(theta);
            auto phi = spherical_phi(wi);
            return vec2(phi * Inv2Pi, 1.0f - theta * InvPi);
        }

        Spectrum Li(const vec3 &wo, const vec2 &) const override {
            auto uv = get_uv(world_to_light.apply_vector(-wo));
            ShadingPoint sp;
            sp.texCoords = uv;
            return color->evaluate(sp);
        }
        LightType get_light_type() const override { return LightType::EDeltaPosition; }
        Float pdf_incidence(const Interaction &ref, const vec3 &wi) const override {
            if (!distribution)
                return 0.0f;
            auto w = world_to_light.apply_vector(wi);
            auto theta = spherical_theta(w);
            auto sinTheta = std::sin(theta);
            auto phi = spherical_phi(w);
            if (sinTheta == 0.0f)
                return 0.0f;
            return distribution->pdf_continuous(Point2f(phi * Inv2Pi, 1.0f - theta * InvPi)) / (2 * Pi * Pi * sinTheta);
        }
        void pdf_emission(const Ray &ray, Float *pdfPos, Float *pdfDir) const override { AKR_PANIC("Not implemented"); }
        void sample_incidence(const vec2 &u, const Interaction &ref, RayIncidentSample *sample,
                              VisibilityTester *tester) const override {
            Float mapPdf = 0.0f;
            if (!distribution) {
                mapPdf = 0.0f;
                sample->pdf = 0.0f;
                sample->I = Spectrum(0);
                return;
            }
            auto uv = distribution->sample_continuous(u, &mapPdf);
            if (mapPdf == 0.0f) {
                sample->I = Spectrum(0);
                sample->pdf = 0.0f;
                return;
            }
            auto theta = (1.0f - uv[1]) * Pi;
            auto phi = uv[0] * 2.0f * Pi;
            Float cosTheta = std::cos(theta);
            Float sinTheta = std::sin(theta);
            sample->wi = light_to_world.apply_vector(spherical_to_xyz(sinTheta, cosTheta, phi));
            if (sinTheta == 0.0f)
                sample->pdf = 0.0f;
            else
                sample->pdf = mapPdf / (2 * Pi * Pi * sinTheta);
            sample->I = color->evaluate(ShadingPoint{uv});
            tester->shadowRay = Ray(ref.p, sample->wi, Eps() / abs(dot(sample->wi, ref.Ng)), Inf);
        }
        void sample_emission(const vec2 &u1, const vec2 &u2, RayEmissionSample *sample) const override {
            AKR_PANIC("Not implemented");
        }
        void commit() {
            Light::commit();
            if (!color) {
                distribution = nullptr;
                return;
            }
            color->commit();
            light_to_world = Transform(transform.matrix4());
            world_to_light = light_to_world.inverse();
            std::vector<Float> v(distribution_map_resolution * distribution_map_resolution);
            AtomicFloat sum(0.0f);
            parallel_for(
                distribution_map_resolution,
                [=, &v, &sum](uint32_t j, uint32_t) {
                    float s = 0;
                    for (size_t i = 0; i < distribution_map_resolution; i++) {
                        vec2 uv(i, j);
                        uv /= vec2(distribution_map_resolution, distribution_map_resolution);
                        ShadingPoint sp{uv};
                        Spectrum L = color->evaluate(sp);
                        v[i + j * distribution_map_resolution] = L.luminance();
                        s += L.luminance();
                    }
                    sum.add(s);
                },
                256);
            distribution =
                std::make_unique<Distribution2D>(&v[0], distribution_map_resolution, distribution_map_resolution);
            _power = sum.value() / v.size() * 4 * Pi * world_radius * world_radius;
        }
    };
    class InfiniteAreaLight : public WorldLightFactory {
        [[refl]] std::shared_ptr<Texture> color;
        [[refl]] TransformManipulator transform;

      public:
        AKR_IMPLS(WorldLightFactory)
        virtual std::shared_ptr<Light> create(const Scene &scene) const {
            return std::make_shared<InfiniteAreaLightImpl>(color, scene.bounds());
        }
    };
#include "generated/InfiniteAreaLight.hpp"
    AKR_EXPORT_PLUGIN(p) {}
} // namespace akari