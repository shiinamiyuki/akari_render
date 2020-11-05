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

#include <akari/core/image.h>
#include <memory>
#include <akari/core/logger.h>
#include <akari/core/parallel.h>
#include <mutex>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

namespace akari {
    using RGBSpectrum = Color<float, 3>;
    class LDRImageWriter : public ImageWriter {
      public:
        bool write(const Image &image, const fs::path &path) override {
            AKR_ASSERT(image.channels() == 3 || image.channels() == 4 || image.channels() == 1);
            const auto ext = path.extension().string();
            auto dimension = image.resolution();
            auto texels = image.data();
            std::vector<uint8_t> buffer(hprod(dimension) * image.channels());
            if (image.channels() >= 3) {
                AKR_ASSERT(image.channel_name(0) == "R");
                AKR_ASSERT(image.channel_name(1) == "G");
                AKR_ASSERT(image.channel_name(2) == "B");
                if (image.channels() >= 4) {
                    AKR_ASSERT(image.channel_name(3) == "A");
                }
            }
            if (image.channels() >= 3) {
                thread::parallel_for(thread::blocked_range<1>(hprod(dimension), 1024), [&](uint32_t i, uint32_t) {
                    auto pixel = static_cast<uint8_t *>(&buffer[i * image.channels()]);
                    auto rgb = linear_to_srgb(load<Color3f>(&texels[i * image.channels()]));
                    rgb = clamp(rgb, vec3(0), vec3(1));
                    for (int comp = 0; comp < 3; comp++) {
                        pixel[comp] = (uint8_t)std::clamp<int>((int)std::round(rgb[comp] * 255.5), 0, 255);
                    }
                });
            } else if (image.channels() == 1) {
                thread::parallel_for(thread::blocked_range<1>(hprod(dimension), 1024), [&](uint32_t i, uint32_t) {
                    auto pixel = static_cast<uint8_t *>(&buffer[i * image.channels()]);
                    auto rgb = linear_to_srgb(load<Color3f>(&texels[i * image.channels()]));
                    rgb = clamp(rgb, vec3(0), vec3(1));
                    pixel[0] = (uint8_t)std::clamp<int>((int)std::round(luminance(rgb) * 255.5), 0, 255);
                });
            }
            if (ext == ".png")
                return stbi_write_png(path.string().c_str(), dimension.x, dimension.y, image.channels(), buffer.data(),
                                      0);
            else if (ext == ".jpg")
                return stbi_write_jpg(path.string().c_str(), dimension.x, dimension.y, image.channels(), buffer.data(),
                                      0);
            return false;
        }
    };

    std::shared_ptr<ImageWriter> ldr_image_writer() { return std::make_shared<LDRImageWriter>(); }

    class LDRImageReader : public ImageReader {
      public:
        std::shared_ptr<Image> read(const fs::path &path) override {
            info("Loading {}", path.string());
            std::shared_ptr<Image> image;
            int x, y, channel;
            auto ext = path.extension().string();
            if (ext == ".hdr") {
                const float *data = stbi_loadf(path.string().c_str(), &x, &y, &channel, 3);
                image = std::make_shared<Image>(rgba_image(ivec2(x, y)));
                thread::parallel_for(image->resolution().y, [=, &image](uint32_t y, uint32_t) {
                    for (int x = 0; x < image->resolution().x; x++) {
                        RGBSpectrum rgb;
                        if (channel == 1) {
                            rgb = RGBSpectrum(data[x + y * image->resolution().x]);
                        } else {
                            rgb[0] = data[3 * (x + y * image->resolution().x) + 0];
                            rgb[1] = data[3 * (x + y * image->resolution().x) + 1];
                            rgb[2] = data[3 * (x + y * image->resolution().x) + 2];
                        }
                        (*image)(x, y, 0) = rgb[0];
                        (*image)(x, y, 1) = rgb[1];
                        (*image)(x, y, 2) = rgb[2];
                        (*image)(x, y, 3) = 1.0;
                    }
                });
                stbi_image_free((void *)data);
            } else {
                const auto *data = stbi_load(path.string().c_str(), &x, &y, &channel, 0);
                image = std::make_shared<Image>(rgba_image(ivec2(x, y)));
                thread::parallel_for(image->resolution().y, [=, &image](uint32_t y, uint32_t) {
                    for (int x = 0; x < image->resolution().x; x++) {
                        RGBSpectrum rgb;
                        if (channel == 1) {
                            rgb = RGBSpectrum((float)data[x + y * image->resolution().x] / 255.0f);
                        } else {
                            rgb[0] = (float)data[channel * (x + y * image->resolution().x) + 0] / 255.0f;
                            rgb[1] = (float)data[channel * (x + y * image->resolution().x) + 1] / 255.0f;
                            rgb[2] = (float)data[channel * (x + y * image->resolution().x) + 2] / 255.0f;
                        }
                        rgb = srgb_to_linear(rgb);
                        float alpha = 1.0;
                        if (channel == 4) {
                            alpha = (float)data[channel * (x + y * image->resolution().x) + 3] / 255.0f;
                        }
                        (*image)(x, y, 0) = rgb[0];
                        (*image)(x, y, 1) = rgb[1];
                        (*image)(x, y, 2) = rgb[2];
                        (*image)(x, y, 3) = alpha;
                    }
                });
                stbi_image_free((void *)data);
            }
            return image;
        }
    };

    std::shared_ptr<ImageReader> default_image_reader() { return std::make_shared<LDRImageReader>(); }
} // namespace akari
