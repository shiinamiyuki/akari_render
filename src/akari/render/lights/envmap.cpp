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
    class AKR_EXPORT InfiniteAreaLight : public Light {
        Transform w2l, l2w;
        static constexpr size_t distribution_map_resolution = 4096;
        std::unique_ptr<Distribution2D> distribution = nullptr;
        Bounds3f world_bounds;
        std::shared_ptr<const Texture> texture = nullptr;
        Float _power = 0.0;

      public:
        static vec2 get_uv(const vec3 &wi) {
            auto theta = spherical_theta(wi);
            // auto sinTheta = std::sin(theta);
            auto phi = spherical_phi(wi);
            return vec2(phi * Inv2Pi, 1.0f - theta * InvPi);
        }
        Float power() const override { return _power; }
        Spectrum Le(const Vec3 &wo, const ShadingPoint &) const override {
            auto uv = get_uv(w2l.apply_vector(-wo));
            ShadingPoint sp;
            sp.texcoords = uv;
            return texture->evaluate(sp);
        }
        Float pdf_incidence(const ReferencePoint &ref, const vec3 &wi) const override {
            if (!distribution)
                return 0.0f;
            auto w = w2l.apply_vector(wi);
            auto theta = spherical_theta(w);
            auto sinTheta = std::sin(theta);
            auto phi = spherical_phi(w);
            if (sinTheta == 0.0f)
                return 0.0f;
            return distribution->pdf_continuous(vec2(phi * Inv2Pi, 1.0f - theta * InvPi)) / (2 * Pi * Pi * sinTheta);
        }
        LightSample sample_incidence(const LightSampleContext &ctx) const override {
            LightSample sample;
            Float mapPdf = 0.0f;
            if (!distribution) {
                mapPdf = 0.0f;
                sample.pdf = 0.0f;
                sample.I = Spectrum(0);
                return sample;
            }
            auto uv = distribution->sample_continuous(ctx.u, &mapPdf);
            if (mapPdf == 0.0f) {
                sample.I = Spectrum(0);
                sample.pdf = 0.0f;
                return sample;
            }
            auto theta = (1.0f - uv[1]) * Pi;
            auto phi = uv[0] * 2.0f * Pi;
            Float cosTheta = std::cos(theta);
            Float sinTheta = std::sin(theta);
            sample.wi = l2w.apply_vector(spherical_to_xyz(sinTheta, cosTheta, phi));
            if (sinTheta == 0.0f)
                sample.pdf = 0.0f;
            else
                sample.pdf = mapPdf / (2 * Pi * Pi * sinTheta);
            sample.I = texture->evaluate(ShadingPoint{uv});
            sample.shadow_ray = Ray(ctx.p, sample.wi, Eps, Inf);
            return sample;
        }
        static std::shared_ptr<InfiniteAreaLight> create(const Scene &scene, const TRSTransform &transform,
                                                         std::shared_ptr<const Texture> texture,
                                                         Allocator<> allocator) {
            auto light = make_pmr_shared<InfiniteAreaLight>(allocator);
            light->l2w = transform();
            light->w2l = light->l2w.inverse();
            info("Updating importance map");
            light->world_bounds = scene.accel->world_bounds();
            auto center = light->world_bounds.centroid();
            auto world_radius = glm::length(center - light->world_bounds.pmin);
            std::vector<Float> v(distribution_map_resolution * distribution_map_resolution);
            AtomicFloat sum(0.0f);
            thread::parallel_for(distribution_map_resolution, [=, &v, &sum](uint32_t j, uint32_t) {
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
            });
            light->texture = texture;
            light->distribution = std::make_unique<Distribution2D>(&v[0], distribution_map_resolution,
                                                                   distribution_map_resolution, Allocator<>());
            light->_power = sum.value() / v.size() * 4 * Pi * world_radius * world_radius;
            return light;
        }
        LightType type() const override { return LightType::Generic; }
    };

    class AKR_EXPORT EnvMapNode final : public LightNode {
      public:
        AKR_SER_CLASS("EnvMap")
        AKR_SER(transform, envmap)
        TRSTransform transform;
        std::shared_ptr<TextureNode> envmap;
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "texture") {
                envmap = resolve_texture(value);
                AKR_ASSERT_THROW(envmap);
            } else if (field == "rotation") {
                transform.rotation = radians(render::load<vec3>(value));
            }
        }
        std::shared_ptr<const Light> create(Allocator<> allocator, const Scene *scene,
                                            const std::optional<Triangle> &triangle) override {
            AKR_CHECK(!triangle.has_value());
            return InfiniteAreaLight::create(*scene, transform, envmap->create_texture(allocator), allocator);
        }
        void finalize() override { envmap->finalize(); }
    };
    AKR_EXPORT_NODE(EnvMap, EnvMapNode);
} // namespace akari::render