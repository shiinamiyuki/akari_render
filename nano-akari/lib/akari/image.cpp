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
#include <OpenImageIO/imageio.h>
#include <akari/image.h>
namespace akari {
    Image array2d_to_rgb(const Array2D<Color3f> &array) {
        Image img = rgb_image(array.dimension());
        thread::parallel_for(array.dimension().y, [&](uint32_t y, uint32_t) {
            for (int x = 0; x < array.dimension().x; x++) {
                auto color = array(x,y);
                img(x, y, 0) = color[0];
                img(x, y, 1) = color[1];
                img(x, y, 2) = color[2];

            }
        });
        return img;
    }
    bool write_hdr(const Image &image, const fs::path &path) {
        AKR_ASSERT(image.channels() == 3 || image.channels() == 4 || image.channels() == 1);
        const auto ext = path.extension().string();
        AKR_ASSERT(ext == ".exr");
        auto dimension = image.resolution();
        const int channels = image.channels();
        using namespace OIIO;
        std::unique_ptr<ImageOutput> out = ImageOutput::create(path.string());
        if (!out)
            return false;
        ImageSpec spec(dimension.x, dimension.y, channels, TypeDesc::FLOAT);
        out->open(path.string(), spec);
        out->write_image(TypeDesc::FLOAT, image.data());
        out->close();
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
        const int channels = image.channels(); // RGB
        using namespace OIIO;
        std::unique_ptr<ImageOutput> out = ImageOutput::create(path.string());
        if (!out)
            return false;
        ImageSpec spec(dimension.x, dimension.y, channels, TypeDesc::UINT8);
        out->open(path.string(), spec);
        out->write_image(TypeDesc::UINT8, buffer.data());
        out->close();
        return true;
    }

    // Image read_generic_image(const fs::path &path){

    // }
} // namespace akari