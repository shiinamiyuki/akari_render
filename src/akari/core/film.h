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
#include <akari/common/buffer.h>

namespace akari {
    AKR_VARIANT struct Pixel {
        AKR_IMPORT_TYPES()
        Spectrum radiance = Spectrum(0);
        Float weight = 0;
    };
    constexpr size_t TileSize = 16;
    AKR_VARIANT struct Tile {
        AKR_IMPORT_TYPES()
        Bounds2i bounds{};
        Point2i _size;
        Buffer<Pixel<C>> pixels;

        explicit Tile(const Bounds2i &bounds)
            : bounds(bounds), _size(bounds.size() + Point2i(2, 2)), pixels(_size.x() * _size.y()) {}

        AKR_XPU auto &operator()(const Point2f &p) {
            auto q = Point2i(floor(p + Point2f(0.5) - Point2f(bounds.pmin) + Point2f(1)));
            return pixels[q.x() + q.y() * _size.x()];
        }

        AKR_XPU const auto &operator()(const Point2f &p) const {
            auto q = Point2i(floor(p + Point2f(0.5) - Point2f(bounds.pmin) + Point2f(1)));
            return pixels[q.x() + q.y() * _size.x()];
        }

        AKR_XPU void add_sample(const Point2f &p, const Spectrum &radiance, Float weight) {
            auto &pix = (*this)(p);
            pix.weight += weight;
            pix.radiance += radiance;
        }
    };
    AKR_VARIANT class Film {
        AKR_IMPORT_TYPES()
        TImage<Spectrum> radiance;
        TImage<Float> weight;

      public:
        Float splatScale = 1.0f;
        explicit Film(const Point2i &dimension) : radiance(dimension), weight(dimension) {}
        Tile<C> tile(const Bounds2i &bounds) { return Tile<C>(bounds); }

        [[nodiscard]] AKR_XPU Point2i resolution() const { return radiance.resolution(); }

        [[nodiscard]] AKR_XPU Bounds2i bounds() const { return Bounds2i{Point2i(0), resolution()}; }
        AKR_XPU void merge_tile(const Tile<C> &tile) {
            const auto lo = max(tile.bounds.pmin - Point2i(1, 1), Point2i(0, 0));
            const auto hi = min(tile.bounds.pmax + Point2i(1, 1), radiance.resolution());
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
                            image(x, y) = std::make_pair(Color<float, 3>(color), 1);
                        } else {
                            image(x, y) = std::make_pair(Color<float, 3>(radiance(x, y)), 1);
                        }
                    }
                },
                1024);
            default_image_writer()->write(image, path, postProcessor);
        }
    };
} // namespace akari
#endif // AKARIRENDER_FILM_H
