// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <akari/image.h>
#include <spdlog/spdlog.h>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImfOutputFile.h>
namespace akari {
    Image array2d_to_rgb(const Array2D<Color3f> &array) {
        Image img = rgb_image(array.dimension());
        thread::parallel_for(array.dimension().y, [&](uint32_t y, uint32_t) {
            for (int x = 0; x < array.dimension().x; x++) {
                auto color = array(x, y);
                img(x, y, 0) = color[0];
                img(x, y, 1) = color[1];
                img(x, y, 2) = color[2];
            }
        });
        return img;
    }
    bool write_hdr(const Image &image, const fs::path &path) {
        auto width = image.resolution()[0];
        // AKR_ASSERT(is_rgb_image(image));
        AKR_ASSERT(path.extension().string() == ".exr");
        Imf::Header header(image.resolution()[0], image.resolution()[1]);
        for (int ch = 0; ch < image.channels(); ch++) {
            header.channels().insert(image.channel_name(ch), Imf::Channel(Imf::FLOAT));
        }
        Imf::OutputFile file(path.string().c_str(), header);
        Imf::FrameBuffer frameBuffer;
        for (int ch = 0; ch < image.channels(); ch++) {
            frameBuffer.insert(image.channel_name(ch),                                // name
                               Imf::Slice(Imf::FLOAT,                                 // type
                                          (char *)(image.data() + ch),                // base
                                          sizeof(float) * image.channels(),           // xStride
                                          sizeof(float) * image.channels() * width)); // yStride
        }
        file.setFrameBuffer(frameBuffer);
        file.writePixels(image.resolution()[1]);
        return true;
    }
    bool write_ldr(const Image &image, const fs::path &path) {
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
            return stbi_write_png(path.string().c_str(), dimension.x, dimension.y, image.channels(), buffer.data(), 0);
        else if (ext == ".jpg")
            return stbi_write_jpg(path.string().c_str(), dimension.x, dimension.y, image.channels(), buffer.data(), 0);
        return false;
    }
    Image read_hdr(const fs::path &path) {
        std::optional<Image> image;
        int x, y, channel;
        auto ext = path.extension().string();
        if (ext == ".hdr") {
            const float *data = stbi_loadf(path.string().c_str(), &x, &y, &channel, 3);
            image = (rgba_image(ivec2(x, y)));
            thread::parallel_for(image->resolution().y, [=, &image](uint32_t y, uint32_t) {
                for (int x = 0; x < image->resolution().x; x++) {
                    Color3f rgb;
                    if (channel == 1) {
                        rgb = Color3f(data[x + y * image->resolution().x]);
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
            spdlog::error("reading {} not supported", ext);
            return rgba_image(ivec2(1));
        }
        return *image;
    }
    Image read_ldr(const fs::path &path) {
        std::optional<Image> image;
        int x, y, channel;
        auto ext = path.extension().string();

        const auto *data = stbi_load(path.string().c_str(), &x, &y, &channel, 0);
        image = (rgba_image(ivec2(x, y)));
        thread::parallel_for(image->resolution().y, [=, &image](uint32_t y, uint32_t) {
            for (int x = 0; x < image->resolution().x; x++) {
                Color3f rgb;
                if (channel == 1) {
                    rgb = Color3f((float)data[x + y * image->resolution().x] / 255.0f);
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

        return *image;
    }
    Image read_generic_image(const fs::path &path) {
        if (path.extension() == ".exr" || path.extension() == ".hdr") {
            return read_hdr(path);
        } else {
            return read_ldr(path);
        }
    }
    bool write_generic_image(const Image &image, const fs::path &path) {
        if (path.extension() == ".exr") {
            return write_hdr(image, path);
        } else {
            return write_ldr(image, path);
        }
    }
} // namespace akari