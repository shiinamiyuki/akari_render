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

#ifndef AKARIRENDER_FILM_H
#define AKARIRENDER_FILM_H

#include <akari/core/fwd.h>
#include <akari/core/image.h>
#include <akari/core/parallel.h>
#include <akari/core/color.h>
#include <akari/core/memory.h>

namespace akari {
    template <class T>
    struct TPixel {
        T radiance = Spectrum(0);
        Float weight = 0;
    };
    constexpr size_t TileSize = 16;
    template <class T>
    struct TTile {

        Bounds2i bounds{};
        ivec2 _size;
        astd::pmr::vector<TPixel<T>> pixels;

        explicit TTile(const Bounds2i &bounds) : bounds(bounds), _size(bounds.size()), pixels(_size.x * _size.y) {}

        auto &operator()(const vec2 &p) {
            auto q = ivec2(floor(p - vec2(bounds.pmin)));
            return pixels[q.x + q.y * _size.x];
        }

        auto &operator()(const ivec2 &p) {
            auto q = ivec2(p - bounds.pmin);
            return pixels[q.x + q.y * _size.x];
        }
        const auto &operator()(const ivec2 &p) const {
            auto q = ivec2(p - bounds.pmin);
            return pixels[q.x + q.y * _size.x];
        }
        const auto &operator()(const vec2 &p) const {
            auto q = ivec2(floor(p - vec2(bounds.pmin)));
            return pixels[q.x + q.y * _size.x];
        }

        void add_sample(const vec2 &p, const Spectrum &radiance, Float weight) {
            auto &pix = (*this)(p);
            pix.weight += weight;
            pix.radiance += radiance;
        }
    };
    template <class T>
    class TFilm {
        TImage<T> radiance;
        TImage<Float> weight;

      public:
        Float splatScale = 1.0f;
        explicit TFilm(const ivec2 &dimension) : radiance(dimension), weight(dimension) {}
        TTile<T> tile(const Bounds2i &bounds) { return TTile<T>(Bounds2i(ivec2(0), resolution()).intersect(bounds)); }
        [[nodiscard]] ivec2 resolution() const { return radiance.resolution(); }

        [[nodiscard]] Bounds2i bounds() const { return Bounds2i{ivec2(0), resolution()}; }
        void merge_tile(const TTile<T> &tile) {
            const auto lo = max(tile.bounds.pmin, ivec2(0, 0));
            const auto hi = min(tile.bounds.pmax, radiance.resolution());
            for (int y = lo.y; y < hi.y; y++) {
                for (int x = lo.x; x < hi.x; x++) {
                    auto &pix = tile(ivec2(x, y));
                    radiance(x, y) += pix.radiance;
                    weight(x, y) += pix.weight;
                }
            }
        }
        template <typename = std::enable_if_t<std::is_same_v<T, Color3f>>>
        RGBAImage to_rgba_image() const {
            RGBAImage image(resolution());
            parallel_for(
                radiance.resolution().y,
                [&](uint32_t y, uint32_t) {
                    for (int x = 0; x < radiance.resolution().x; x++) {
                        if (weight(x, y) != 0) {
                            auto color = (radiance(x, y)) / weight(x, y);
                            image(x, y) = RGBA(color, 1.0);
                        } else {
                            image(x, y) = RGBA(radiance(x, y), 1.0);
                        }
                    }
                },
                1024);
            return image;
        }
        TImage<T> to_image() const {
            TImage<T> image(resolution());
            parallel_for(
                radiance.resolution().y,
                [&](uint32_t y, uint32_t) {
                    for (int x = 0; x < radiance.resolution().x; x++) {
                        if (weight(x, y) != 0) {
                            auto color = (radiance(x, y)) / weight(x, y);
                            image(x, y) = color;
                        } else {
                            image(x, y) = radiance(x, y);
                        }
                    }
                },
                1024);
            return image;
        }

        template <typename = std::enable_if_t<std::is_same_v<T, Color3f>>>
        void write_image(const fs::path &path, const PostProcessor &postProcessor = GammaCorrection()) const {
            RGBAImage image = to_rgba_image();
            default_image_writer()->write(image, path, postProcessor);
        }
    };
    using Film = TFilm<Spectrum>;
    using Tile = TTile<Spectrum>;
} // namespace akari
#endif // AKARIRENDER_FILM_H
