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
#include <akari/render/scenegraph.h>
#include <akari/render/texture.h>
#include <akari/core/color.h>
#include <akari/core/resource.h>
#include <akari/core/image.hpp>
#include <akari/render/common.h>
namespace akari::render {
    class ConstantTexture : public Texture {
        Spectrum value;

      public:
        ConstantTexture(Spectrum v) : value(v) {}
        Spectrum evaluate(const ShadingPoint &sp) const override { return value; }
        Float integral() const override { return luminance(value); }
    };
    class ConstantTextureNode final : public TextureNode {
        Spectrum value;

      public:
        ConstantTextureNode(Spectrum v) : value(v) {}
        Texture *create_texture(Allocator<> *allocator) override {
            return allocator->new_object<ConstantTexture>(value);
        }
    };

    class ImageTexture : public Texture {
        RGBAImage::View image;

      public:
        ImageTexture(RGBAImage::View image) : image(image) {}
        Spectrum evaluate(const ShadingPoint &sp) const override {
            vec2 texcoords = sp.texcoords;
            vec2 tc = glm::mod(texcoords, vec2(1.0f));
            tc.y = 1.0f - tc.y;
            return image(tc).rgb;
        }
        Float integral() const override {
            Float I = 0;
            for (int i = 0; i < image.resolution().x * image.resolution().y; i++) {
                I += luminance(image.data()[i].rgb);
            }
            return I / (image.resolution().x * image.resolution().y);
        }
    };
    class ImageTextureNode final : public TextureNode {
        std::shared_ptr<RGBAImage> image;

      public:
        ImageTextureNode() = default;
        ImageTextureNode(const fs::path &path) {
            auto res = resource_manager()->load_path<ImageResource>(path);
            if (res) {
                image = res.extract_value()->image();
            } else {
                auto err = res.extract_error();
                error("error loading {}: {}", path.string(), err.what());
                throw std::runtime_error("Error loading image");
            }
        }
        Texture *create_texture(Allocator<> *allocator) override {
            return allocator->new_object<ImageTexture>(image->view());
        }
    };
    AKR_EXPORT std::shared_ptr<TextureNode> create_constant_texture() {
        return std::make_shared<ConstantTextureNode>(Spectrum());
    }
    AKR_EXPORT std::shared_ptr<TextureNode> create_image_texture() { return std::make_shared<ImageTextureNode>(); }
    AKR_EXPORT std::shared_ptr<TextureNode> create_constant_texture_rgb(const RGB &value) {
        return std::make_shared<ConstantTextureNode>(value);
    }
    AKR_EXPORT std::shared_ptr<TextureNode> create_image_texture(const std::string &path) {
        return std::make_shared<ImageTextureNode>(path);
    }
} // namespace akari::render