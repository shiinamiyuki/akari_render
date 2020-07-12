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

#include "spectrum.h"
#include <akari/core/image.hpp>
#include <akari/core/parallel.h>

namespace akari {
    struct Pixel {
        Spectrum radiance = Spectrum(0);
        Float weight = 0;
    };
    struct SplatPixel {
        std::array<AtomicFloat, 3> color;
        static_assert(sizeof(AtomicFloat) == sizeof(float));
        float _padding{};
    };
    static_assert(sizeof(SplatPixel) == 4 * sizeof(float));
    const size_t TileSize = 16;
    struct Tile {
        Bounds2i bounds{};
        ivec2 _size;
        std::vector<Pixel> pixels;

        explicit Tile(const Bounds2i &bounds)
            : bounds(bounds), _size(bounds.size() + ivec2(2, 2)), pixels(_size.x * _size.y) {}

        auto &operator()(const vec2 &p) {
            auto q = ivec2(floor(p + vec2(0.5) - vec2(bounds.p_min) + vec2(1)));
            return pixels[q.x + q.y * _size.x];
        }

        const auto &operator()(const vec2 &p) const {
            auto q = ivec2(floor(p + vec2(0.5) - vec2(bounds.p_min) + vec2(1)));
            return pixels[q.x + q.y * _size.x];
        }

        void add_sample(const vec2 &p, const Spectrum &radiance, Float weight) {
            auto &pix = (*this)(p);
            pix.weight += weight;
            pix.radiance += radiance;
        }
    };

    class Film {
        TImage<Spectrum> radiance;
        TImage<float> weight;
        TImage<SplatPixel> splat;

      public:
        Float splatScale = 1.0f;
        explicit Film(const ivec2 &dimension) : radiance(dimension), weight(dimension), splat(dimension) {}
        Tile tile(const Bounds2i &bounds) { return Tile(bounds); }

        [[nodiscard]] ivec2 resolution() const { return radiance.resolution(); }

        [[nodiscard]] Bounds2i bounds() const { return Bounds2i{ivec2(0), resolution()}; }
        void merge_tile(const Tile &tile) {
            const auto lo = max(tile.bounds.p_min - ivec2(1, 1), ivec2(0, 0));
            const auto hi = min(tile.bounds.p_max + ivec2(1, 1), radiance.resolution());
            for (int y = lo.y; y < hi.y; y++) {
                for (int x = lo.x; x < hi.x; x++) {
                    auto &pix = tile(vec2(x, y));
                    radiance(x, y) += pix.radiance;
                    weight(x, y) += pix.weight;
                }
            }
        }

        void write_image(const fs::path &path, const PostProcessor &postProcessor = GammaCorrection()) const {
            RGBAImage image(resolution());
            parallel_for(
                    radiance.resolution().y,
                    [&](uint32_t y, uint32_t) {
                        for (int x = 0; x < radiance.resolution().x; x++) {
                            Spectrum s =
                                    Spectrum(splat(x, y).color[0], splat(x, y).color[1], splat(x, y).color[2]) *
                                    splatScale;
                            if (weight(x, y) != 0) {
                                auto color = (radiance(x, y) + s) / weight(x, y);
                                image(x, y) = std::make_pair(color, 1);
                            } else {
                                image(x, y) = std::make_pair(s, 1);
                            }
                        }
                    },
                    1024);
            default_image_writer()->write(image, path, postProcessor);
        }

        void add_splat(const Spectrum &L, const vec2 &p) {
            ivec2 ip = ivec2(p);
            splat(ip).color[0].add(L[0]);
            splat(ip).color[1].add(L[1]);
            splat(ip).color[2].add(L[2]);
        }
    };
} // namespace akari
#endif // AKARIRENDER_FILM_H
