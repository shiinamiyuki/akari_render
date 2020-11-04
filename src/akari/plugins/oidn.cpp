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

#include <OpenImageDenoise/oidn.hpp>
#include <akari/render/denoiser.h>

namespace akari::render {
    class OIDNDenoiser final : public Denoiser {
        oidn::DeviceRef device;

      public:
        OIDNDenoiser() {
            device = oidn::newDevice();
            device.commit();
        }
        void add_aov_requests(RenderInput &inputs) {
            inputs.requested_aovs["normal"].required_variance = false;
            inputs.requested_aovs["albedo"].required_variance = false;
        }
        std::optional<RGBAImage> denoise3(RGBAImage &color, RGBAImage &albedo, RGBAImage &normal) {
            oidn::FilterRef filter;
            filter = device.newFilter("RT");
            auto res = color.resolution();
            filter.setImage("color", color.data(), oidn::Format::Float3, res.x, res.y, 0, sizeof(RGBA));
            filter.setImage("albedo", albedo.data(), oidn::Format::Float3, res.x, res.y, 0, sizeof(RGBA));
            filter.setImage("normal", normal.data(), oidn::Format::Float3, res.x, res.y, 0, sizeof(RGBA));
            auto output = RGBAImage(res);
            filter.setImage("output", output.data(), oidn::Format::Float3, res.x, res.y, 0, sizeof(RGBA));
            filter.set("hdr", true);
            filter.commit();
            filter.execute();

            // Check for errors
            const char *errorMessage;
            if (device.getError(errorMessage) != oidn::Error::None) {
                error("OIDNDenoiser: {}", errorMessage);
                return std::nullopt;
            }
            return output;
        }
        std::optional<RGBAImage> denoise(const Scene *scene, RenderOutput &aov) override {
            oidn::FilterRef filter;
            RGBAImage color = aov.aovs["color"].value->to_rgba_image();
            RGBAImage albedo = aov.aovs["albedo"].value->to_rgba_image();
            RGBAImage normal = aov.aovs["normal"].value->to_rgba_image();
            RGBAImage first_hit_normal = aov.aovs["first_hit_normal"].value->to_rgba_image();
            RGBAImage first_hit_albedo = aov.aovs["first_hit_albedo"].value->to_rgba_image();
            std::optional<RGBAImage> filtered_normal = denoise3(normal, first_hit_albedo, first_hit_normal);
            std::optional<RGBAImage> filtered_albedo = denoise3(albedo, first_hit_albedo, first_hit_normal);
            if (!filtered_albedo || !filtered_normal) {
                return std::nullopt;
            }
            normal = aov.aovs["normal"].value->to_rgba_image();
            auto output = denoise3(color, *filtered_albedo, *filtered_normal);
            info("oidn denoise complete");
            return output;
        }
    };
    AKR_EXPORT_PLUGIN(OIDNDenoiser, OIDNDenoiser, Denoiser);
} // namespace akari::render