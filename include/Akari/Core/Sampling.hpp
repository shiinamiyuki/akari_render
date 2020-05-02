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

#ifndef AKARIRENDER_SAMPLING_HPP
#define AKARIRENDER_SAMPLING_HPP
#include <Akari/Core/Math.h>
#include <algorithm>

namespace Akari {
    inline vec2 concentric_disk_sampling(const vec2 &u) {
        vec2 uOffset = 2.f * u - vec2(1, 1);
        if (uOffset.x == 0 && uOffset.y == 0)
            return vec2(0, 0);

        Float theta, r;
        if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
            r = uOffset.x;
            theta = Pi4 * (uOffset.y / uOffset.x);
        } else {
            r = uOffset.y;
            theta = Pi2 - Pi4 * (uOffset.x / uOffset.y);
        }
        return r * vec2(std::cos(theta), std::sin(theta));
    }

    inline vec3 cosine_hemisphere_sampling(const vec2 &u) {
        auto uv = concentric_disk_sampling(u);
        auto r = dot(uv, uv);
        auto h = std::sqrt(std::max(0.0f, 1 - r));
        return vec3(uv.x, h, uv.y);
    }
    inline Float cosine_hemisphere_pdf(Float cosTheta) { return cosTheta * InvPi; }
    inline Float uniform_sphere_pdf() { return 1.0f / (4 * Pi); }
    inline vec3 uniform_sphere_sampling(const vec2 &u) {
        Float z = 1 - 2 * u[0];
        Float r = std::sqrt(std::max((Float)0, (Float)1 - z * z));
        Float phi = 2 * Pi * u[1];
        return vec3(r * std::cos(phi), r * std::sin(phi), z);
    }
} // namespace Akari
#endif // AKARIRENDER_SAMPLING_HPP
