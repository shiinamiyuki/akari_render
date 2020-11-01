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
    class AreaLight final : public Light {
      public:
        Triangle triangle;
        std::shared_ptr<const Texture> color;
        bool double_sided = false;
        vec3 ng;
        AreaLight() = default;
        AreaLight(Triangle triangle, std::shared_ptr<const Texture> color, bool double_sided)
            : triangle(triangle), color(color), double_sided(double_sided) {
            ng = triangle.ng();
        }
        Spectrum Le(const Vec3 &wo, const ShadingPoint &sp) const override {
            bool face_front = dot(wo, ng) > 0.0;
            if (double_sided || face_front) {
                return color->evaluate(sp);
            }
            return Spectrum(0.0);
        }
        Float pdf_incidence(const ReferencePoint &ref, const vec3 &wi) const override {
            Ray ray(ref.p, wi);
            auto hit = triangle.intersect(ray);
            if (!hit) {
                return 0.0f;
            }
            Float SA = triangle.area() * (-glm::dot(wi, triangle.ng())) / (hit->first * hit->first);
            return 1.0f / SA;
        }
        LightSample sample_incidence(const LightSampleContext &ctx) const override {
            auto coords = uniform_sample_triangle(ctx.u);
            auto p = triangle.p(coords);
            LightSample sample;
            sample.ng = triangle.ng();
            sample.wi = p - ctx.p;
            auto dist_sqr = dot(sample.wi, sample.wi);
            sample.wi /= sqrt(dist_sqr);
            sample.I = color->evaluate(triangle.texcoord(coords));
            sample.pdf = dist_sqr / max(Float(0.0), -dot(sample.wi, sample.ng)) / triangle.area();
            sample.shadow_ray = Ray(p, -sample.wi, Eps / std::abs(dot(sample.wi, sample.ng)),
                                    sqrt(dist_sqr) * (Float(1.0f) - ShadowEps));
            return sample;
        }
        Float power() const override { return triangle.area() * color->integral(); }
    };

    class AreaLightNode final : public LightNode {
        std::shared_ptr<TextureNode> color;

      public:
        // triangle is given if the light is attached to a triangle
        std::shared_ptr<const Light> create(Allocator<> allocator, const Scene *scene,
                                            const std::optional<Triangle> &triangle) override {
            AKR_CHECK(triangle.has_value());
            return make_pmr_shared<const AreaLight>(allocator, *triangle, color->create_texture(allocator), false);
        }
         void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color") {
                color = resolve_texture(value);
                AKR_ASSERT_THROW(color);
            }
        }
    };
    AKR_EXPORT_NODE(AreaLight, AreaLightNode);
} // namespace akari::render