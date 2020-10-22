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

#include <akari/core/parallel.h>
#include <akari/core/distribution.h>
#include <akari/render/light.h>
#include <akari/render/scene.h>

namespace akari::render {
    std::unique_ptr<InfiniteAreaLight> InfiniteAreaLight::create(const Scene &scene, const TRSTransform &transform,
                                                                 const Texture *texture) {
        auto distribution_map_resolution = InfiniteAreaLight::distribution_map_resolution;
        auto light = std::make_unique<InfiniteAreaLight>();
        light->l2w = transform();
        light->w2l = light->l2w.inverse();
        info("Updating importance map");
        light->world_bounds = scene.accel->world_bounds();
        auto center = light->world_bounds.centroid();
        auto world_radius = glm::length(center - light->world_bounds.pmin);
        std::vector<Float> v(distribution_map_resolution * distribution_map_resolution);
        AtomicFloat sum(0.0f);
        parallel_for(
            distribution_map_resolution,
            [=, &v, &sum](uint32_t j, uint32_t) {
                float s = 0;
                for (size_t i = 0; i < distribution_map_resolution; i++) {
                    vec2 uv(i, j);
                    uv /= vec2(distribution_map_resolution, distribution_map_resolution);
                    ShadingPoint sp(uv);
                    Spectrum L = texture->evaluate(sp);
                    v[i + j * distribution_map_resolution] = luminance(L);
                    s += luminance(L);
                }
                sum.add(s);
            },
            256);
        light->texture = texture;
        light->distribution =
            std::make_unique<Distribution2D>(&v[0], distribution_map_resolution, distribution_map_resolution);
        light->_power = sum.value() / v.size() * 4 * Pi * world_radius * world_radius;
        return light;
    }
} // namespace akari::render