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

#include <akari/core/fwd.h>
#include <akari/core/math.h>
namespace akari {
    template <typename Scalar, int N>
    struct Color : Vector<Scalar, N> {
        using Base = Vector<Scalar, N>;
        using Base::Base;
        using value_t = Scalar;
        static constexpr size_t size = N;
        Color(const Base &v) : Base(v) {}
#define AKR_COLOR_OP(op)                                                                                               \
    Color operator op(const Color &rhs) const { return Color(Base(*this) op Base(rhs)); }                              \
    Color operator op(Scalar rhs) const { return Color(Base(*this) op Base(rhs)); }                                    \
    friend Color operator op(Scalar lhs, const Color &rhs) { return Color(Base(lhs) op Base(rhs)); }                   \
    Color &operator op##=(const Color &rhs) {                                                                          \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }                                                                                                                  \
    Color &operator op##=(Scalar rhs) {                                                                                \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }
        AKR_COLOR_OP(+) AKR_COLOR_OP(-) AKR_COLOR_OP(*) AKR_COLOR_OP(/)
#undef AKR_COLOR_OP
    };
    template <typename Scalar, int N>
    Color<Scalar, N> clamp_zero(const Color<Scalar, N> &in) {
        Color<Scalar, N> c;
        for (int i = 0; i < N; i++) {
            auto x = in[i];
            if (isnan(x)) {
                x = 0;
            } else {
                x = max(Scalar(0.0f), x);
            }
            c[i] = x;
        }
        return c;
    }
    template <typename Scalar, int N>
    bool is_black(const Color<Scalar, N> &color) {
        return !foldl(color, false, [](bool acc, Scalar f) { return acc || (f > 0.0f); });
    }

    template <typename Scalar>
    Color<Scalar, 3> linear_to_srgb(const Color<Scalar, 3> &L) {
        using Color3f = Color<Scalar, 3>;
        return select(glm::lessThan(L, Color3f(0.0031308)), L * 12.92,
                      Float(1.055) * glm::pow(L, Vec3(1.0f / 2.4f)) - Float(0.055));
    }
    template <typename Scalar>
    Color<Scalar, 3> srgb_to_linear(const Color<Scalar, 3> &S) {
        using Color3f = Color<Scalar, 3>;
        return select(glm::lessThan(S, 0.04045), S / 12.92, glm::pow((S + 0.055) / 1.055), Vec3(2.4));
    }

    using Color3f = Color<Float, 3>;

    inline Float luminance(const Color3f &rgb) { return dot(rgb, Vec3(0.2126, 0.7152, 0.0722)); }

    template <typename T, int N>
    struct vec_trait<Color<T, N>> {
        using value_type = T;
        static constexpr int size = N;
    };
} // namespace akari
