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
#include <akari/common/config.h>
#include <akari/common/diagnostic.h>
#include "detail/macro.h"
namespace akari {
    // AkariRender needs a bit of retargeting capability
    // like different types of Spectrum
    // float vs double
    // however vectorization is not supported
    template <typename T, size_t N, int packed> constexpr int compute_padded_size() {
        if constexpr (!std::is_fundamental_v<T>) {
            return N;
        }
        if constexpr (packed || N <= 2) {
            return N;
        }
        if constexpr (sizeof(T) == 1) {
            // round to 128 bits
            return (N + 15u) & ~15u;
        } else if constexpr (sizeof(T) == 2) {
            // round to 128 bits
            return (N + 7u) & ~7u;
        } else if constexpr (sizeof(T) == 4) {
            // round to 128 bits
            return (N + 3u) & ~3u;
        } else if constexpr (sizeof(T) == 8) {
            // round to 128 bits
            return (N + 1u) & ~1u;
        } else {
            return N;
        }
    }
    template <typename T, size_t N, int packed> constexpr int compute_align() {
        if constexpr (!std::is_fundamental_v<T>) {
            return alignof(T);
        }
        if constexpr (packed || N <= 2) {
            return alignof(T);
        }
        return 128 / 32;
    }

#define AKR_VARIANT template <typename Float, typename Spectrum>
    template <typename T, size_t N, int packed = 0> struct alignas(compute_align<T, N, packed>()) Array;
    template <typename Float, int N> struct Vector;
    template <typename Float, int N> struct Point;
    template <typename Float, int N> struct Normal;
    template <typename Float, int N> struct Matrix;
    template <typename Float, int N> struct Color;

    template <typename Float> struct Transform;
    template <typename Vector> struct Frame;
    template <typename Point> struct BoundingBox;

    template <typename T> struct value_ { using type = T; };
    template <typename T, int N> struct value_<Array<T, N>> { using type = T; };
    template <typename T, int N> struct value_<Vector<T, N>> { using type = T; };
    template <typename T, int N> struct value_<Point<T, N>> { using type = T; };
    template <typename T, int N> struct value_<Normal<T, N>> { using type = T; };
    template <typename T> using value_t = typename value_<T>::type;
    template <typename T, typename S> struct replace_scalar_ { using type = S; };
    template <typename T, int N, typename S> struct replace_scalar_<Vector<T, N>, S> { using type = Vector<S, N>; };
    template <typename T, int N, typename S> struct replace_scalar_<Point<T, N>, S> { using type = Point<S, N>; };
    template <typename T, int N, typename S> struct replace_scalar_<Normal<T, N>, S> { using type = Normal<S, N>; };
    template <typename T, int N, typename S> struct replace_scalar_<Matrix<T, N>, S> { using type = Matrix<S, N>; };
    template <typename T, typename S> using replace_scalar_t = typename replace_scalar_<T, S>::type;

    template <typename T> struct array_size { static constexpr size_t value = 1; };
    template <typename T, int N> struct array_size<Point<T, N>> { static constexpr size_t value = N; };
    template <typename T, int N> struct array_size<Array<T, N>> { static constexpr size_t value = N; };
    template <typename T, int N> struct array_size<Vector<T, N>> { static constexpr size_t value = N; };
    template <typename T, int N> struct array_size<Normal<T, N>> { static constexpr size_t value = N; };
    template <typename T> constexpr size_t array_size_v = array_size<T>::value;

    template <typename T> struct is_array : std::false_type {};
    template <typename T, int N> struct is_array<Array<T, N>> : std::true_type {};
    template <typename T, int N> struct is_array<Point<T, N>> : std::true_type {};
    template <typename T, int N> struct is_array<Vector<T, N>> : std::true_type {};
    template <typename T, int N> struct is_array<Normal<T, N>> : std::true_type {};
    template <typename T> constexpr size_t is_array_v = is_array<T>::value;
    template <typename T> struct is_integer : std::is_integral<T> {};
    template <typename T> struct is_float : std::is_floating_point<T> {};
    template <typename T> constexpr static bool is_integer_v = is_integer<T>::value;
    template <typename T> constexpr static bool is_float_v = is_float<T>::value;
    // template <typename T>
    // using int32_array_t = replace_scalar_t<T, int32_t>;
    // template <typename T>
    // using uint32_array_t = replace_scalar_t<T, uint32_t>;
    // template <typename T>
    // using int64_array_t = replace_scalar_t<T, uint64_t>;
    // template <typename T>
    // using uint64_array_t = replace_scalar_t<T, uint64_t>;
    // template <typename T>
    // using uint32_array_t = replace_scalar_t<T, uint32_t>;
    // template <typename T>
    // using float32_array_t = replace_scalar_t<T, float>;
    // template <typename T>
    // using float64_array_t = replace_scalar_t<T, double>;

    template <typename Float> struct CoreAliases {
        using Int8 = int8_t;
        using Int32 = int32_t;
        using UInt32 = uint32_t;
        using Int64 = int64_t;
        using UInt64 = uint64_t;
        using Float32 = float;
        using Float64 = double;

        using Color1f = Color<Float, 1>;
        using Color3f = Color<Float, 3>;

        using Vector1i = Vector<Int32, 1>;
        using Vector2i = Vector<Int32, 2>;
        using Vector3i = Vector<Int32, 3>;
        using Vector4i = Vector<Int32, 4>;

        using Vector1u = Vector<UInt32, 1>;
        using Vector2u = Vector<UInt32, 2>;
        using Vector3u = Vector<UInt32, 3>;
        using Vector4u = Vector<UInt32, 4>;

        using Vector1f = Vector<Float, 1>;
        using Vector2f = Vector<Float, 2>;
        using Vector3f = Vector<Float, 3>;
        using Vector4f = Vector<Float, 4>;

        using Vector1d = Vector<Float64, 1>;
        using Vector2d = Vector<Float64, 2>;
        using Vector3d = Vector<Float64, 3>;
        using Vector4d = Vector<Float64, 4>;

        using Point1i = Point<Int32, 1>;
        using Point2i = Point<Int32, 2>;
        using Point3i = Point<Int32, 3>;
        using Point4i = Point<Int32, 4>;

        using Point1u = Point<UInt32, 1>;
        using Point2u = Point<UInt32, 2>;
        using Point3u = Point<UInt32, 3>;
        using Point4u = Point<UInt32, 4>;

        using Point1f = Point<Float, 1>;
        using Point2f = Point<Float, 2>;
        using Point3f = Point<Float, 3>;
        using Point4f = Point<Float, 4>;

        using Point1d = Point<Float64, 1>;
        using Point2d = Point<Float64, 2>;
        using Point3d = Point<Float64, 3>;
        using Point4d = Point<Float64, 4>;

        using Normal3f = Normal<Float, 3>;
        using Normal3d = Normal<Float64, 3>;

        using Matrix2f = Matrix<Float, 2>;
        using Matrix2d = Matrix<Float64, 2>;
        using Matrix3f = Matrix<Float, 3>;
        using Matrix3d = Matrix<Float64, 3>;
        using Matrix4f = Matrix<Float, 4>;
        using Matrix4d = Matrix<Float64, 4>;

        using Frame3f = Frame<Vector3f>;
        using Transform3f = Transform<Float>;

        using Bounds2i = BoundingBox<Point2i>;
        using Bounds2f = BoundingBox<Point2f>;
        using Bounds3f = BoundingBox<Point3f>;
    };

#define AKR_IMPORT_CORE_TYPES_PREFIX(Float_, prefix)                                                                   \
    using prefix##CoreAliases = akari::CoreAliases<Float_>;                                                            \
    using prefix##Int8 = typename prefix##CoreAliases::Int8;                                                           \
    using prefix##Int32 = typename prefix##CoreAliases::Int32;                                                         \
    using prefix##UInt32 = typename prefix##CoreAliases::UInt32;                                                       \
    using prefix##Int64 = typename prefix##CoreAliases::Int64;                                                         \
    using prefix##UInt64 = typename prefix##CoreAliases::UInt64;                                                       \
    using prefix##Float32 = typename prefix##CoreAliases::Float32;                                                     \
    using prefix##Float64 = typename prefix##CoreAliases::Float64;                                                     \
    using prefix##Vector1i = typename prefix##CoreAliases::Vector1i;                                                   \
    using prefix##Vector2i = typename prefix##CoreAliases::Vector2i;                                                   \
    using prefix##Vector3i = typename prefix##CoreAliases::Vector3i;                                                   \
    using prefix##Vector4i = typename prefix##CoreAliases::Vector4i;                                                   \
    using prefix##Vector1u = typename prefix##CoreAliases::Vector1u;                                                   \
    using prefix##Vector2u = typename prefix##CoreAliases::Vector2u;                                                   \
    using prefix##Vector3u = typename prefix##CoreAliases::Vector3u;                                                   \
    using prefix##Vector4u = typename prefix##CoreAliases::Vector4u;                                                   \
    using prefix##Vector1f = typename prefix##CoreAliases::Vector1f;                                                   \
    using prefix##Vector2f = typename prefix##CoreAliases::Vector2f;                                                   \
    using prefix##Vector3f = typename prefix##CoreAliases::Vector3f;                                                   \
    using prefix##Vector4f = typename prefix##CoreAliases::Vector4f;                                                   \
    using prefix##Vector1d = typename prefix##CoreAliases::Vector1d;                                                   \
    using prefix##Vector2d = typename prefix##CoreAliases::Vector2d;                                                   \
    using prefix##Vector3d = typename prefix##CoreAliases::Vector3d;                                                   \
    using prefix##Vector4d = typename prefix##CoreAliases::Vector4d;                                                   \
    using prefix##Point1i = typename prefix##CoreAliases::Point1i;                                                     \
    using prefix##Point2i = typename prefix##CoreAliases::Point2i;                                                     \
    using prefix##Point3i = typename prefix##CoreAliases::Point3i;                                                     \
    using prefix##Point4i = typename prefix##CoreAliases::Point4i;                                                     \
    using prefix##Point1u = typename prefix##CoreAliases::Point1u;                                                     \
    using prefix##Point2u = typename prefix##CoreAliases::Point2u;                                                     \
    using prefix##Point3u = typename prefix##CoreAliases::Point3u;                                                     \
    using prefix##Point4u = typename prefix##CoreAliases::Point4u;                                                     \
    using prefix##Point1f = typename prefix##CoreAliases::Point1f;                                                     \
    using prefix##Point2f = typename prefix##CoreAliases::Point2f;                                                     \
    using prefix##Point3f = typename prefix##CoreAliases::Point3f;                                                     \
    using prefix##Point4f = typename prefix##CoreAliases::Point4f;                                                     \
    using prefix##Point1d = typename prefix##CoreAliases::Point1d;                                                     \
    using prefix##Point2d = typename prefix##CoreAliases::Point2d;                                                     \
    using prefix##Point3d = typename prefix##CoreAliases::Point3d;                                                     \
    using prefix##Point4d = typename prefix##CoreAliases::Point4d;                                                     \
    using prefix##Normal3f = typename prefix##CoreAliases::Normal3f;                                                   \
    using prefix##Normal3d = typename prefix##CoreAliases::Normal3d;                                                   \
    using prefix##Matrix2f = typename prefix##CoreAliases::Matrix2f;                                                   \
    using prefix##Matrix2d = typename prefix##CoreAliases::Matrix2d;                                                   \
    using prefix##Matrix3f = typename prefix##CoreAliases::Matrix3f;                                                   \
    using prefix##Matrix3d = typename prefix##CoreAliases::Matrix3d;                                                   \
    using prefix##Matrix4f = typename prefix##CoreAliases::Matrix4f;                                                   \
    using prefix##Matrix4d = typename prefix##CoreAliases::Matrix4d;                                                   \
    using prefix##Frame3f = typename prefix##CoreAliases::Frame3f;                                                     \
    using prefix##Transform3f = typename prefix##CoreAliases::Transform3f;                                             \
    using prefix##Color1f = typename prefix##CoreAliases::Color1f;                                                     \
    using prefix##Color3f = typename prefix##CoreAliases::Color3f;                                                     \
    using prefix##Bounds2f = typename prefix##CoreAliases::Bounds2f;                                                   \
    using prefix##Bounds3f = typename prefix##CoreAliases::Bounds3f;                                                   \
    using prefix##Bounds2i = typename prefix##CoreAliases::Bounds2i;

#define AKR_IMPORT_CORE_TYPES() AKR_IMPORT_CORE_TYPES_PREFIX(Float, )
#define AKR_IMPORT_CORE_TYPES_WITH(fl)                                                                                 \
    using Float = fl;                                                                                                  \
    AKR_IMPORT_CORE_TYPES()

    AKR_VARIANT struct Ray;
    AKR_VARIANT class Film;
    AKR_VARIANT struct Pixel;
    AKR_VARIANT struct Tile;
    AKR_VARIANT class Material;
    AKR_VARIANT class BSDFClosure;
    AKR_VARIANT class Scene;
    AKR_VARIANT struct sampling;
    AKR_VARIANT struct microfacet;

    template <typename Float_, typename Spectrum_> struct RenderAliases {
        using Float = Float_;
        using Spectrum = Spectrum_;
        using Ray = Ray<Float, Spectrum>;
        using Film = akari::Film<Float, Spectrum>;
        using Tile = akari::Tile<Float, Spectrum>;
        using Pixel = akari::Pixel<Float, Spectrum>;
        using Material = akari::Material<Float, Spectrum>;
        using BSDFClosure = akari::BSDFClosure<Float, Spectrum>;
        using sampling = akari::sampling<Float, Spectrum>;
        using microfacet = akari::microfacet<Float, Spectrum>;
        using Scene = akari::Scene<Float, Spectrum>;
    };
#define AKR_IMPORT_BASIC_RENDER_TYPES()                                                                                \
    AKR_IMPORT_CORE_TYPES()                                                                                            \
    using RenderAliases = akari::RenderAliases<Float, Spectrum>;                                                       \
    using sampling = typename RenderAliases::sampling;                                                                 \
    using microfacet = typename RenderAliases::microfacet;                                                             \
    using Scene = typename RenderAliases::Scene;

#define AKR_IMPORT_TYPES(...)                                                                                          \
    AKR_IMPORT_BASIC_RENDER_TYPES()                                                                                    \
    AKR_USING_TYPES(RenderAliases, __VA_ARGS__)

} // namespace akari