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
#include <cstdint>
#include <cstddef>
namespace akari {
#define AKR_VARIANT template <typename Float, typename Spectrum>

    template <typename Float, int N> struct Vector;
    template <typename Float, int N> struct Point;
    template <typename Float, int N> struct Normal;
    template <typename Float> struct Matrix4;
    template <typename T, typename S> struct replace_scalar_ { using type = S; };
    template <typename T, int N, typename S> struct replace_scalar_<Vector<T, N>, S> { using type = Vector<S, N>; };
    template <typename T, int N, typename S> struct replace_scalar_<Point<T, N>, S> { using type = Point<S, N>; };
    template <typename T, int N, typename S> struct replace_scalar_<Normal<T, N>, S> { using type = Normal<S, N>; };
    template <typename T, typename S> struct replace_scalar_<Matrix4<T>, S> { using type = Matrix4<S>; };
    template <typename T, typename S> using replace_scalar_t = typename replace_scalar_<T, S>::type;
    template <typename T> using int32_array_t = replace_scalar_t<T, int32_t>;
    template <typename T> using uint32_array_t = replace_scalar_t<T, uint32_t>;
    template <typename T> using int64_array_t = replace_scalar_t<T, uint64_t>;
    template <typename T> using uint64_array_t = replace_scalar_t<T, uint64_t>;
    template <typename T> using uint32_array_t = replace_scalar_t<T, uint32_t>;
    template <typename T> using float32_array_t = replace_scalar_t<T, float>;
    template <typename T> using float64_array_t = replace_scalar_t<T, double>;

    AKR_VARIANT struct CoreAliases {
        using Int8 = replace_scalar_t<Float, int8_t>;
        using Int32 = int32_array_t<Float>;
        using UInt32 = uint32_array_t<Float>;
        using Int64 = int64_array_t<Float>;
        using UInt64 = uint64_array_t<Float>;
        using Float32 = float32_array_t<Float>;
        using Float64 = float64_array_t<Float>;

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

        // using Matrix4f = 
    };

} // namespace akari