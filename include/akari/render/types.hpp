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

#ifndef AKARIRENDER_TYPES_HPP
#define AKARIRENDER_TYPES_HPP
#include <akari/core/math.h>
namespace akari {
    template <typename T, typename U> struct convert_to { using type = U; };

    template <typename T, size_t N, typename U> struct convert_to<simd_array<T, N>, U> {
        using type = simd_array<U, N>;
    };
    template <typename T, typename U> using convert_t = typename convert_to<T, U>::type;


    template <typename Float> struct Ray;
    template <typename Float> struct TIntersection;
    template <typename Float> struct TTriangle;
    template <typename Float> struct TShadingPoint;
    template <typename Float, typename Spectrum> class BSDF;
    template <typename Float, typename Spectrum> class BSDFComponent;
    template <typename Float, typename Spectrum> class Material;
    template <typename Float, typename Spectrum> class Integrator;
#define AKR_BASIC_TYPES()                                                                                              \
    using Scalar = scalar_t<Float>;                                                                                    \
    using Vector3f = vec<3, Float>;                                                                                    \
    using Vector2f = vec<2, Float>;                                                                                    \
    using Int = convert_t<Float, int>;                                                                                 \
    using Vector2i = vec<2, Int>;                                                                                      \
    using Ray = TRay<Float>;                                                                                           \
    using Intersection = TIntersection<Float>;                                                                         \
    using Triangle = TTriangle<Float>;
    using ShadingPoint = TShadingPoint<float>;

} // namespace akari

#endif // AKARIRENDER_TYPES_HPP
