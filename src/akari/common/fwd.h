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
#include <cstdint>
#include <cstddef>
#include <akari/common/def.h>
#include <akari/common/config.h>
#include <akari/common/diagnostic.h>
#include <akari/common/astd.h>

namespace akari {
    // AkariRender needs a bit of retargeting capability
    // like different types of Spectrum
    // float vs double
    // however vectorization is not supported
    template <typename T, int N, int packed>
    constexpr int compute_align() {
        if constexpr (!std::is_fundamental_v<T>) {
            return alignof(T);
        }
        if constexpr (packed || N <= 2) {
            return alignof(T);
        }
        if constexpr (sizeof(T) == 1) {
            return 4;
        }
        if constexpr (sizeof(T) == 2) {
            return 4;
        }
        if constexpr (sizeof(T) == 4) {
            return 16;
        }
        // align to 128 bits (16 bytes)
        return (alignof(T) + 15u) & ~15u;
    }
    template <typename Float_, typename Spectrum_>
    struct Config {
        using Float = Float_;
        using Spectrum = Spectrum_;
    };
#define AKR_VARIANT template <class C>
    template <typename T, int N, int packed = 0>
    struct alignas(compute_align<T, N, packed>()) Array;
    template <typename Float, int N>
    struct Color;
    template <typename Float, int N>
    struct Matrix;
    template <typename Float>
    struct Transform;
    template <typename Vector>
    struct Frame;
    template <typename Point>
    struct BoundingBox;

    template <typename T>
    struct SOA;
    template <typename T>
    struct value_ {
        using type = T;
    };
    template <typename T, int N, int P>
    struct value_<Array<T, N, P>> {
        using type = T;
    };
    template <typename T, int N>
    struct value_<Color<T, N>> {
        using type = T;
    };
    template <typename T>
    using value_t = typename value_<T>::type;
    template <typename T, typename S>
    struct replace_scalar_ {
        static_assert(std::is_fundamental_v<T>);
        using type = S;
    };
    template <typename T, int N, int P, typename S>
    struct replace_scalar_<Array<T, N, P>, S> {
        using type = Array<S, N>;
    };
    template <typename T, typename S>
    struct replace_scalar_<SOA<T>, S> {
        using type = SOA<S>;
    };

    template <typename T, int N, typename S>
    struct replace_scalar_<Color<T, N>, S> {
        using type = Color<S, N>;
    };
    template <typename T, int N, typename S>
    struct replace_scalar_<Matrix<T, N>, S> {
        using type = Matrix<S, N>;
    };
    template <typename T, typename S>
    using replace_scalar_t = typename replace_scalar_<T, S>::type;

    template <typename T>
    struct array_size {
        static constexpr size_t value = 1;
    };
    template <typename T, int N, int P>
    struct array_size<Array<T, N, P>> {
        static constexpr size_t value = N;
    };
    template <typename T, int N>
    struct array_size<Color<T, N>> {
        static constexpr size_t value = N;
    };
    template <typename T>
    constexpr size_t array_size_v = array_size<T>::value;
    template <typename Value, int N>
    using Vector = Array<Value, N>;
    template <typename Value, int N>
    using Point = Array<Value, N>;
    template <typename Value, int N>
    using Normal = Array<Value, N>;
    template <typename T>
    struct is_array : std::false_type {};
    template <typename T, int N, int P>
    struct is_array<Array<T, N, P>> : std::true_type {};

    template <typename T, int N>
    struct is_array<Color<T, N>> : std::true_type {};
    template <typename T>
    constexpr size_t is_array_v = is_array<T>::value;
    template <typename T>
    struct is_integer : std::is_integral<T> {};
    template <typename T>
    struct is_float : std::is_floating_point<T> {};
    template <typename T>
    constexpr static bool is_integer_v = is_integer<T>::value;
    template <typename T>
    constexpr static bool is_float_v = is_float<T>::value;
    template <typename T>
    using int8_array_t = replace_scalar_t<T, int8_t>;
    template <typename T>
    using int32_array_t = replace_scalar_t<T, int32_t>;
    template <typename T>
    using uint32_array_t = replace_scalar_t<T, uint32_t>;
    template <typename T>
    using int64_array_t = replace_scalar_t<T, uint64_t>;
    template <typename T>
    using uint64_array_t = replace_scalar_t<T, uint64_t>;
    template <typename T>
    using uint32_array_t = replace_scalar_t<T, uint32_t>;
    template <typename T>
    using float32_array_t = replace_scalar_t<T, float>;
    template <typename T>
    using float64_array_t = replace_scalar_t<T, double>;

    using bool1 = Array<bool, 1>;
    using bool2 = Array<bool, 2>;
    using bool3 = Array<bool, 3>;
    using bool4 = Array<bool, 4>;
    using bool8 = Array<bool, 8>;

    using char1 = Array<int8_t, 1>;
    using char2 = Array<int8_t, 2>;
    using char3 = Array<int8_t, 3>;
    using char4 = Array<int8_t, 4>;
    using char8 = Array<int8_t, 8>;

    using uchar1 = Array<uint8_t, 1>;
    using uchar2 = Array<uint8_t, 2>;
    using uchar3 = Array<uint8_t, 3>;
    using uchar4 = Array<uint8_t, 4>;
    using uchar8 = Array<uint8_t, 8>;

    using float1 = Array<float, 1>;
    using float2 = Array<float, 2>;
    using float3 = Array<float, 3>;
    using float4 = Array<float, 4>;
    using float8 = Array<float, 8>;

    using int1 = Array<int, 1>;
    using int2 = Array<int, 2>;
    using int3 = Array<int, 3>;
    using int4 = Array<int, 4>;
    using int8 = Array<int, 8>;

    using uint1 = Array<uint32_t, 1>;
    using uint2 = Array<uint32_t, 2>;
    using uint3 = Array<uint32_t, 3>;
    using uint4 = Array<uint32_t, 4>;
    using uint8 = Array<uint32_t, 8>;

    using ulong1 = Array<uint64_t, 1>;
    using ulong2 = Array<uint64_t, 2>;
    using ulong3 = Array<uint64_t, 3>;
    using ulong4 = Array<uint64_t, 4>;
    using ulong8 = Array<uint64_t, 8>;

    using long1 = Array<int64_t, 1>;
    using long2 = Array<int64_t, 2>;
    using long3 = Array<int64_t, 3>;
    using long4 = Array<int64_t, 4>;
    using long8 = Array<int64_t, 8>;

    using double1 = Array<double, 1>;
    using double2 = Array<double, 2>;
    using double3 = Array<double, 3>;
    using double4 = Array<double, 4>;
    using double8 = Array<double, 8>;

    using matrix3 = Matrix<float, 3>;
    using matrix4 = Matrix<float, 4>;
    using frame3f = Frame<float3>;
    using transform3f = Transform<float>;
    using bounds2i = BoundingBox<int2>;
    using bounds2f = BoundingBox<float2>;
    using bounds3f = BoundingBox<float3>;

    using color1f = Color<float, 1>;
    using color2f = Color<float, 2>;
    using color3f = Color<float, 3>;
    using color4f = Color<float, 4>;

#define AKR_IMPORT_CORE_TYPES()                                                                                        \
    using Array1f = Array<Float, 1>;                                                                                   \
    using Array2f = Array<Float, 2>;                                                                                   \
    using Array3f = Array<Float, 3>;                                                                                   \
    using Array4f = Array<Float, 4>;                                                                                   \
    using Float1 = Array<Float, 1>;                                                                                    \
    using Float2 = Array<Float, 2>;                                                                                    \
    using Float3 = Array<Float, 3>;                                                                                    \
    using Float4 = Array<Float, 4>;                                                                                    \
    using Color1f = Color<Float, 1>;                                                                                   \
    using Color2f = Color<Float, 2>;                                                                                   \
    using Color3f = Color<Float, 3>;                                                                                   \
    using Color4f = Color<Float, 4>;                                                                                   \
    using Matrix3f = Matrix<Float, 3>;                                                                                 \
    using Matrix4f = Matrix<Float, 4>;                                                                                 \
    using Frame3f = Frame<Float3>;                                                                                     \
    using Transform3f = Transform<Float>;                                                                              \
    using Bounds2i = BoundingBox<int2>;                                                                                \
    using Bounds2f = BoundingBox<Float2>;                                                                              \
    using Bounds3f = BoundingBox<Float3>;

#define AKR_IMPORT_CORE_TYPES_WITH(fl)                                                                                 \
    using Float = fl;                                                                                                  \
    AKR_IMPORT_CORE_TYPES()

    AKR_VARIANT struct Ray;
    AKR_VARIANT class Film;
    AKR_VARIANT struct Pixel;
    AKR_VARIANT struct Tile;
    AKR_VARIANT class Material;
    AKR_VARIANT class Camera;
    AKR_VARIANT class BSDFClosure;
    AKR_VARIANT class Scene;
    AKR_VARIANT class Sampler;
    AKR_VARIANT struct sampling;
    AKR_VARIANT struct microfacet;
    AKR_VARIANT struct bsdf;
    AKR_VARIANT struct CameraSample;
#define AKR_IMPORT_TYPES()                                                                                             \
    using Float = typename C::Float;                                                                                   \
    using Spectrum = typename C::Spectrum;                                                                             \
    using Ray3f = Ray<C>;                                                                                              \
    AKR_IMPORT_CORE_TYPES()
#ifdef AKR_ENABLE_EMBREE
#    define akari_enable_embree true
#else
#    define akari_enable_embree false
#endif

    template <typename T>
    using device_ptr = T *;
} // namespace akari