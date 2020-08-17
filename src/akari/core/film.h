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

#include <akari/common/fwd.h>
#include <akari/core/image.hpp>
#include <akari/core/parallel.h>


namespace akari {
    AKR_VARIANT struct Pixel {
        Spectrum radiance = Spectrum(0);
        Float weight = 0;
    };
    constexpr size_t TileSize = 16;
    AKR_VARIANT struct Tile {
        AKR_IMPORT_TYPES(Pixel)
        Bounds2i bounds{};
        Point2i _size;
        std::vector<APixel> pixels;

        explicit Tile(const Bounds2i &bounds)
            : bounds(bounds), _size(bounds.size() + Point2f(2, 2)), pixels(_size.x() * _size.y()) {}

        auto &operator()(const Point2f &p) {
            auto q = Point2i(floor(p + Point2f(0.5) - Point2f(bounds.p_min) + Point2f(1)));
            return pixels[q.x() + q.y() * _size.x()];
        }

        const auto &operator()(const Point2f &p) const {
            auto q = Point2i(floor(p + Point2f(0.5) - Point2f(bounds.p_min) + Point2f(1)));
            return pixels[q.x() + q.y() * _size.x()];
        }

        void add_sample(const Point2f &p, const Spectrum &radiance, Float weight) {
            auto &pix = (*this)(p);
            pix.weight += weight;
            pix.radiance += radiance;
        }
    };

    AKR_VARIANT class Film {
        TImage<Spectrum> radiance;
        TImage<Float> weight;
        AKR_IMPORT_TYPES(Tile, Pixel)
      public:
        Float splatScale = 1.0f;
        explicit Film(const Point2i &dimension) : radiance(dimension), weight(dimension) {}
        ATile tile(const Bounds2i &bounds) { return Tile(bounds); }

        [[nodiscard]] Point2i resolution() const { return radiance.resolution(); }

        [[nodiscard]] Bounds2i bounds() const { return Bounds2i{Point2i(0), resolution()}; }
        void merge_tile(const ATile &tile) {
            const auto lo = max(tile.bounds.p_min - Point2i(1, 1), Point2i(0, 0));
            const auto hi = min(tile.bounds.p_max + Point2i(1, 1), radiance.resolution());
            for (int y = lo.y(); y < hi.y(); y++) {
                for (int x = lo.x(); x < hi.x(); x++) {
                    auto &pix = tile(Point2f(x, y));
                    radiance(x, y) += pix.radiance;
                    weight(x, y) += pix.weight;
                }
            }
        }

        void write_image(const fs::path &path, const PostProcessor &postProcessor = GammaCorrection()) const {
            RGBAImage image(resolution());
            parallel_for(
                    radiance.resolution().y(),
                    [&](uint32_t y, uint32_t) {
                        for (int x = 0; x < radiance.resolution().x(); x++) {
                            if (weight(x, y) != 0) {
                                auto color = (radiance(x, y)) / weight(x, y);
                                image(x, y) = std::make_pair(color, 1);
                            } else {
                                image(x, y) = std::make_pair(radiance(x, y), 1);
                            }
                        }
                    },
                    1024);
            default_image_writer()->write(image, path, postProcessor);
        }
    };
} // namespace akari
#endif // AKARIRENDER_FILM_H
