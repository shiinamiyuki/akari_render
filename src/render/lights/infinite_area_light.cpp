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
        std::shared_ptr<Texture> strength;
        std::unique_ptr<Distribution2D> distribution;
        TransformManipulator transform;
        Transform world_to_light, light_to_world;
        const size_t distribution_map_resolution = 4096;

      public:
        AKR_IMPLS(Light)

        Float pdf_incidence(const Interaction &ref, const vec3 &wi) const override { return 0; }
        void pdf_emission(const Ray &ray, Float *pdfPos, Float *pdfDir) const override { AKR_PANIC("Not implemented"); }
        void commit() {
            Light::commit();
            if (!color) {
                distribution = nullptr;
                return;
            }
            light_to_world = Transform(transform.matrix4());
            world_to_light = light_to_world.inverse();
            std::vector<Float> v(distribution_map_resolution * distribution_map_resolution);
            parallel_for(distribution_map_resolution, [=, &v](uint32_t j, uint32_t) {
                for (size_t i = 0; i < distribution_map_resolution; i++) {
                    vec2 uv(i, j);
                    uv /= vec2(distribution_map_resolution, distribution_map_resolution);
                    uv[1] = 1.0f - uv[1];
                    ShadingPoint sp{uv};
                    Spectrum L = color->evaluate(sp) * strength->evaluate(sp);
                    v[i + j * distribution_map_resolution] = L.luminance();
                }
            });
            distribution =
                std::make_unique<Distribution2D>(&v[0], distribution_map_resolution, distribution_map_resolution);
        }
    };
    class InfiniteAreaLight : public WorldLightFactory {
        [[refl]] std::shared_ptr<Texture> color;
        [[refl]] std::shared_ptr<Texture> strength;
        [[refl]] TransformManipulator transform;

      public:
        virtual std::shared_ptr<Light> create(const Scene &scene) const {
            return create_infinite_area_light(nullptr, scene.bounds());
        }
    };
#include "generated/InfiniteAreaLight.hpp"
    AKR_EXPORT_PLUGIN(p) {}
} // namespace akari