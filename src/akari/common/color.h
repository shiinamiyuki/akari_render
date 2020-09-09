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

#include <akari/common/fwd.h>
#include <akari/common/math.h>
namespace akari {
    template <typename Float, int N>
    struct Color : Array<Float, N> {
        using Base = Array<Float, N>;
        using Base::Base;
        using value_t = Float;
        static constexpr size_t size = N;
        AKR_ARRAY_IMPORT(Base, Color)
        Color clamp_zero() const {
            Color c;
            for (int i = 0; i < N; i++) {
                auto x = (*this)[i];
                if (std::isnan(x)) {
                    x = 0;
                } else {
                    x = max(Float(0.0f), x);
                }
                c[i] = x;
            }
            return c;
        }
        bool is_black() const {
            return !reduce(*this, [](bool acc, Float f) { return acc || (f > 0.0f); });
        }
    };

    template <typename Float>
    Color<Float, 3> linear_to_srgb(const Color<Float, 3> &L) {
        using Color3f = Color<Float, 3>;
        return select(L < 0.0031308, L * 12.92, 1.055 * pow(L, Float(1.0f / 2.4f)) - 0.055);
    }
    template <typename Float>
    Color<Float, 3> srgb_to_linear(const Color<Float, 3> &S) {
        using Color3f = Color<Float, 3>;
        return select(S < 0.04045, S / 12.92, pow((S + 0.055) / 1.055), Float(2.4));
    }
} // namespace akari
