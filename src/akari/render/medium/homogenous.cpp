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

#include <akari/render/media.h>
namespace akari::render {
    class HomogenousMedium : public Medium {
      public:
        Float g = 0.0;
        Spectrum sigma_t, sigma_s, sigma_a;
        Float density_a, density_s;
        Spectrum tr(const Ray &ray, Sampler *sampler) const override {
            Float d = ray.tmax * length(ray.d);
            auto tau = sigma_t * -d;
            auto tr = exp(tau);
            return tr;
        }
        MediumSample sample(const Ray &ray, Sampler *sampler, Allocator<> allocator) const override {
            MediumSample sample;
            int channel = std::min<int>(Spectrum::size - 1, (int)(sampler->next1d() * Spectrum::size));
            Float dist = -std::log(1 - sampler->next1d()) / sigma_t[channel];
            Float tmax = ray.tmax * length(ray.d);
            Float t = std::min(dist, tmax);
            bool sampled_medium = t < tmax;
            if (sampled_medium) {
                sample.phase = allocator.new_object<HGPhaseFunction>(g);
                sample.p = ray.o + t * ray.d / length(ray.d);
            }
            Spectrum tr = exp(sigma_t * -t);
            Spectrum density = sampled_medium ? (sigma_t * tr) : tr;
            Float pdf = 0.0;
            for (uint32_t i = 0; i < Spectrum::size; i++) {
                pdf += density[i];
            }
            pdf /= Float(Spectrum::size);
            sample.tr = sampled_medium ? tr * sigma_s / pdf : tr / pdf;
            return sample;
        }
    };
    class HomogenousMediumNode final : public MediumNode {

      public:
        AKR_SER_CLASS("HomogenousMediumNode")
        Spectrum color_a, color_s;
        Float density_a = 1.0f;
        Float density_s = 1.0f;
        Float g;
        std::shared_ptr<const Medium> create_medium(Allocator<> allocator) override {
            auto m = make_pmr_shared<HomogenousMedium>(allocator);
            m->g = g;
            Spectrum sigma_a = color_a * density_a;
            Spectrum sigma_s = color_s * density_s;
            Spectrum sigma_t = sigma_a + sigma_s;
            m->sigma_t = sigma_t;
            m->sigma_s = sigma_s;
            m->sigma_a = sigma_a;
            m->density_a = density_a;
            m->density_s = density_s;
            return m;
        }
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "color_a") {
                color_a = render::load<Color3f>(value);
            } else if (field == "color_s") {
                color_a = render::load<Color3f>(value);
            } else if (field == "density_a") {
                density_a = value.get<double>().value();
            } else if (field == "density_s") {
                density_s = value.get<double>().value();
            }
        }
    };
    AKR_EXPORT_NODE(HomogenousMedium, HomogenousMediumNode)
} // namespace akari::render