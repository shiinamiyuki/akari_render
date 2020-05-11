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
#include <akari/core/platform.h>
#include <akari/generated/fwd.h>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
namespace akari {
    template <typename T, typename U> struct replace_ { using type = U; };

    template <typename T, size_t N, typename U> struct replace_<simd_array<T, N>, U> { using type = simd_array<U, N>; };
    template <typename T, typename U> using replace_scalar_t = typename replace_<T, U>::type;

#define __AKR_USING_TYPE(_r, _data, type) using type = typename Collection::type;
#define AKR_USE_TYPES(...)                BOOST_PP_SEQ_FOR_EACH(__AKR_USING_TYPE, _, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
#define AKR_USE_MATH_CONSTANTS()                                                                                       \
    static constexpr Float Pi             = constants<Float>::Pi;                                                      \
    static constexpr Float Pi2            = constants<Float>::Pi2;                                                     \
    static constexpr Float Pi4            = constants<Float>::Pi4;                                                     \
    static constexpr Float InvPi          = constants<Float>::InvPi;                                                   \
    static constexpr Float Inv4Pi         = constants<Float>::Inv4Pi;                                                  \
    static constexpr Float Inf            = constants<Float>::Inf;                                                     \
    static constexpr Float MaxFloat       = constants<Float>::MaxFloat;                                                \
    static constexpr Float MinFloat       = constants<Float>::MinFloat;                                                \
    static constexpr Float MachineEpsilon = constants<Float>::MachineEpsilon;

#define AKR_BASIC_TYPES()                                                                                              \
    AKR_USE_MATH_CONSTANTS()                                                                                           \
    using Collection = akari::Collection<Float, Spectrum>;                                                             \
    using Scalar     = scalar_t<Float>;                                                                                \
    using Vector3f   = vec<3, Float>;                                                                                  \
    using Vector2f   = vec<2, Float>;                                                                                  \
    using Int        = replace_scalar_t<Float, int>;                                                                   \
    using Vector2i   = vec<2, Int>;                                                                                    \
    using Bool       = replace_scalar_t<Float, bool>;

#define AKR_GEOMETRY_TYPES()                                                                                           \
    AKR_USE_TYPES(Ray, Vertex, Triangle, SurfaceSample, Intersection, Interaction, SurfaceInteraction,                 \
                  EndPointInteraction, VolumeInteraction)
#define AKR_COMPONENT_TYPES()                                                                                          \
    AKR_USE_TYPES(BSDF, BSDFComponent, Material, Integrator, Texture, EndPoint, Camera, Sampler)
#define AKR_UTIL_TYPES() AKR_USE_TYPES(Emission, MaterialSlot)

} // namespace akari

#endif // AKARIRENDER_TYPES_HPP
