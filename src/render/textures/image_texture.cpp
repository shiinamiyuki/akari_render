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

#include <akari/core/image.hpp>
#include <akari/core/parallel.h>
#include <akari/core/plugin.h>
#include <akari/plugins/image_texture.h>
#include <akari/core/resource.h>
#include <akari/core/logger.h>
namespace akari {
    class ImageTexture : public Texture {
        std::shared_ptr<RGBAImage> image;
        [[refl]] fs::path path;
        Float average = 0;

      public:
        ImageTexture() = default;
        ImageTexture(const fs::path &path) : path(path) {}
        AKR_IMPLS(Texture)
        void commit() override {
            auto exp = resource_manager()->load_path<ImageResource>(path);
            if(!exp.has_value()){
                error("cannot load {}: {}", path.string(),exp.error().what());
                return;
            }
            auto tmp = (*exp)->image();
            if (tmp != image) {
                image = tmp;
                AtomicFloat sum(0);
                parallel_for(
                    image->resolution().y,
                    [=, &sum](uint32_t y, uint32_t) {
                        for (int x = 0; x < image->resolution().x; x++) {
                            auto color = (*image)(x, y);
                            Spectrum rgb = Spectrum(color.x, color.y, color.z);
                            sum.add(rgb.luminance());
                        }
                    },
                    32);
                average = sum.value() / float(image->resolution().x * image->resolution().y);
            }
        }
        Float average_luminance() const override { return average; }
        Spectrum evaluate(const ShadingPoint &sp) const override {
            auto texCoords = mod(sp.texCoords, vec2(1));
            return (*image)(texCoords);
        }
    };
#include "generated/ImageTexture.hpp"
    AKR_EXPORT_PLUGIN(p) {
    }

    std::shared_ptr<Texture> create_image_texture(const fs::path &path) { return std::make_shared<ImageTexture>(path); }
} // namespace akari
