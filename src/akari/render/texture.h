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

#pragma once
#include <akari/core/variant.h>
#include <akari/core/color.h>
#include <akari/core/image.h>
#include <akari/render/scenegraph.h>

namespace akari::render {
    struct ShadingPoint {
        ShadingPoint() = default;
        ShadingPoint(const vec2 &tc) : texcoords(tc) {}
        vec2 texcoords;
    };

    class Texture {
      public:
        virtual Spectrum evaluate(const ShadingPoint &sp) const = 0;
        virtual Float integral() const = 0;
        virtual Float tr(const ShadingPoint &sp) const = 0;
        virtual ~Texture() = default;
    };
    class TextureNode : public SceneGraphNode {
      public:
        virtual std::shared_ptr<const Texture> create_texture(Allocator<>) = 0;
    };
    class ConstantTexture : public Texture {
        Spectrum value;
        Float alpha = 0;

      public:
        ConstantTexture(Spectrum v, Float alpha) : value(v), alpha(alpha) {}
        Spectrum evaluate(const ShadingPoint &sp) const override { return value; }
        Float integral() const override { return luminance(value); }
        Float tr(const ShadingPoint &sp) const override { return std::max<Float>(0.0, 1.0f - alpha); }
    };
    class ImageTexture : public Texture {
        std::shared_ptr<RGBAImage> image;
        mutable ObjectCache<Float> integral_;

      public:
        ImageTexture(std::shared_ptr<RGBAImage> image) : image(image) {}
        Spectrum evaluate(const ShadingPoint &sp) const override {
            vec2 texcoords = sp.texcoords;
            vec2 tc = glm::mod(texcoords, vec2(1.0f));
            tc.y = 1.0f - tc.y;
            return (*image)(tc).rgb;
        }
        Float integral() const override {
            return integral_.get_cached_or([&] {
                Float I = 0;
                for (int i = 0; i < image->resolution().x * image->resolution().y; i++) {
                    I += luminance(image->data()[i].rgb);
                }
                return I / (image->resolution().x * image->resolution().y);
            });
        }
        Float tr(const ShadingPoint &sp) const override {
            vec2 texcoords = sp.texcoords;
            vec2 tc = glm::mod(texcoords, vec2(1.0f));
            tc.y = 1.0f - tc.y;
            return std::max<Float>(0.0, 1.0f - (*image)(tc).alpha);
        }
    };

    AKR_EXPORT std::shared_ptr<TextureNode> create_constant_texture();
    AKR_EXPORT std::shared_ptr<TextureNode> create_image_texture();
    AKR_EXPORT std::shared_ptr<TextureNode> create_constant_texture_rgba(const RGBA &value);
    AKR_EXPORT std::shared_ptr<TextureNode> create_constant_texture_rgb(const RGB &value);
    AKR_EXPORT std::shared_ptr<TextureNode> create_image_texture(const std::string &path);
} // namespace akari::render