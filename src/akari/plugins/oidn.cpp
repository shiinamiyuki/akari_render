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
        std::optional<RGBAImage> denoise(const Scene *scene, AOV &aov) override {
            oidn::FilterRef filter;
            filter = device.newFilter("RT");
            auto res = aov.aovs.at("color").resolution();
            filter.setImage("color", aov.aovs.at("color").data(), oidn::Format::Float3, res.x, res.y, 0, sizeof(RGBA));
            filter.setImage("albedo", aov.aovs.at("albedo").data(), oidn::Format::Float3, res.x, res.y, 0,
                            sizeof(RGBA));
            filter.setImage("normal", aov.aovs.at("normal").data(), oidn::Format::Float3, res.x, res.y, 0,
                            sizeof(RGBA));
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
            info("oidn denoise complete");
            return output;
        }
    };
    AKR_EXPORT_PLUGIN(OIDNDenoiser, OIDNDenoiser, Denoiser);
} // namespace akari::render