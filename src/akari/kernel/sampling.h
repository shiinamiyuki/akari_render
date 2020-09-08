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

#ifndef AKARIRENDER_SAMPLING_HPP
#define AKARIRENDER_SAMPLING_HPP
#include <akari/common/math.h>
#include <algorithm>

namespace akari {
    AKR_VARIANT
    struct sampling {
        AKR_IMPORT_TYPES()
        static inline Point2f concentric_disk_sampling(const Point2f &u) {
            Point2f uOffset = 2.f * u - Point2f(1, 1);
            if (uOffset.x() == 0 && uOffset.y() == 0)
                return Point2f(0, 0);

            Float theta, r;
            if (std::abs(uOffset.x()) > std::abs(uOffset.y())) {
                r = uOffset.x();
                theta = Constants<Float>::Pi4 * (uOffset.y() / uOffset.x());
            } else {
                r = uOffset.y();
                theta = Constants<Float>::Pi2 - Constants<Float>::Pi4 * (uOffset.x() / uOffset.y());
            }
            return r * Point2f(std::cos(theta), std::sin(theta));
        }

        static inline Vector3f cosine_hemisphere_sampling(const Point2f &u) {
            auto uv = concentric_disk_sampling(u);
            auto r = dot(uv, uv);
            auto h = std::sqrt(std::max(Float(0.0f), Float(1.0f - r)));
            return Vector3f(uv.x(), h, uv.y());
        }
        static inline Float cosine_hemisphere_pdf(Float cosTheta) { return cosTheta * Constants<Float>::InvPi; }
        static inline Float uniform_sphere_pdf() { return 1.0f / (4 * Constants<Float>::Pi); }
        static inline Vector3f uniform_sphere_sampling(const Point2f &u) {
            Float z = 1 - 2 * u[0];
            Float r = std::sqrt(max((Float)0, (Float)1 - z * z));
            Float phi = 2 * Constants<Float>::Pi * u[1];
            return Vector3f(r * cos(phi), r * sin(phi), z);
        }
        static inline Point2f uniform_sample_triangle(const Point2f &u) {
            Float su0 = std::sqrt(u[0]);
            Float b0 = 1 - su0;
            Float b1 = u[1] * su0;
            return Point2f(b0, b1);
        }
    };
} // namespace akari
#endif // AKARIRENDER_SAMPLING_HPP
