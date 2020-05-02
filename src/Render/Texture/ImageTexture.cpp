// MIT License
//
// Copyright (c) 2019 椎名深雪
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

#include <Akari/Core/Image.hpp>
#include <Akari/Core/Parallel.h>
#include <Akari/Core/Plugin.h>
#include <Akari/Plugins/ImageTexture.h>
namespace Akari {
    class ImageTexture : public Texture {
        std::shared_ptr<RGBAImage> image;
        fs::path path;
        Float average = 0;

      public:
        ImageTexture() = default;
        ImageTexture(const fs::path &path) : path(path) {}
        AKR_SER(path)
        AKR_DECL_COMP(ImageTexture, "ImageTexture")
        void commit() override {
            auto loader = GetImageLoader();
            auto tmp = loader->Load(path);
            if (tmp != image) {
                image = tmp;
                AtomicFloat sum(0);
                ParallelFor(
                    image->Dimension().y,
                    [=, &sum](uint32_t y, uint32_t) {
                        for (int x = 0; x < image->Dimension().x; x++) {
                            auto color = (*image)(x, y);
                            Spectrum rgb = Spectrum(color.x, color.y, color.z);
                            sum.add(rgb.Luminance());
                        }
                    },
                    32);
                average = sum.value() / float(image->Dimension().x * image->Dimension().y);
            }
        }
        Float average_luminance() const override { return average; }
        Spectrum evaluate(const ShadingPoint &sp) const override {
            auto texCoords = mod(sp.texCoords, vec2(1));
            return (*image)(texCoords);
        }
    };
    AKR_EXPORT_COMP(ImageTexture, "Texture")

    std::shared_ptr<Texture> CreateImageTexture(const fs::path &path) { return std::make_shared<ImageTexture>(path); }
} // namespace Akari
