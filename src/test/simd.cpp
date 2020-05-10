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

#include <akari/core/application.h>
#include <akari/core/image.hpp>
#include <akari/core/profiler.hpp>
#include <akari/core/simd.hpp>
#include <complex>
using namespace akari;

template <size_t N> Packed<int, N> mandelbrot(const Packed<float, N> &x0, const Packed<float, N> &y0) {
    using Mask = Packed<bool, N>;
    static const int max_iter = 100;
    Mask active = true;
    auto x = x0;
    auto y = y0;
    Packed<int, N> iters = max_iter;
    for (int i = 0; any(active) && i < max_iter; i++) {
        auto outside = x * x + y * y >= 4.0f;
        masked(iters, active && outside) = i;
        masked(active, outside) = false;
        auto xtmp = x * x - y * y;
        auto ytmp = 2.0f * x * y;
        y = ytmp + y0;
        x = xtmp + x0;
    }
    return iters;
}

template <typename F> void tiled_for(uint32_t nx, uint32_t ny, F &&f) {
    size_t constexpr simd_lanes = 8u;
    uint32_t max_m8 = nx & ~(simd_lanes - 1u);
    Packed<uint32_t, simd_lanes> inc;
    for (size_t i = 0; i < simd_lanes; i++) {
        inc[i] = i;
    }
    for (uint32_t y = 0u; y < ny; y++) {
        for (uint32_t x = 0u; x < max_m8; x += simd_lanes) {
            Packed<uint32_t, simd_lanes> packed_x = inc + x, packed_y(y);
            f(simd_tag<simd_lanes>, packed_x, packed_y);
        }
        for (uint32_t x = max_m8; x < nx; x++) {
            f(simd_tag<1>, x, y);
        }
    }
}
template <typename F> void scalar_for(uint32_t nx, uint32_t ny, F &&f) {
    for (uint32_t y = 0u; y < ny; y++) {
        for (uint32_t x = 0u; x < nx; x++) {
            f(simd_tag<1>, x, y);
        }
    }
}
int main() {
    Application app;
    //    Packed<float *, 32> v;
    //    Packed<float, 32> a, b;
    //    for (int i = 0; i < 32; i++) {
    //        a[i] = 2 * i + 1;
    //        b[i] = 3 * i + 2;
    //    }
    //    a = a + b;
    //    auto mask = array_operator_lt<float, 32>::apply(a, b);
    //    for (int i = 0; i < 32; i++) {
    //        printf("%f %f %d\n", a[i], b[i], mask[i]);
    //    }
    //    auto c = select(~((a < 100.0f) & (a > 50.0f)), a, b);
    //    a[mask] = b;
    //    simd_array<float, 32> &ref = a;
    //    ref = b;
    //    for (int i = 0; i < 32; i++) {
    //        printf("%f %f %f %d\n", a[i], b[i], c[i], mask[i]);
    //    }
    //    Packed<float, 128> x(2.3);
    //    for (int i = 0; i < 128; i++) {
    //        x[i] = 3.1415 * i / 128.0;
    //    }
    //    auto y = sin(x);
    //    auto z = cos(x);
    //    for (int i = 0; i < 128; i++) {
    //        printf("%lf %lf %lf %lf\n", sin(x[i]), y[i], cos(x[i]), z[i]);
    //    }
    RGBAImage image(ivec2(500, 500));
    auto f = [&](auto &&tag, auto &&px, auto &&py) {
        constexpr size_t lanes = std::decay_t<decltype(tag)>::lanes;
        using F = Packed<float, lanes>;
        F x, y;
        x = F(px) / 500.0f;
        y = F(py) / 500.0f;
        x = 2.0f * x - 1.0f;
        y = 2.0f * y - 1.0f;
        x *= 1.5f;
        y *= 1.5f;
        x -= 0.5f;
        auto result = F(mandelbrot<lanes>(x, y)) / 200.0f;
        if constexpr (lanes == 1) {
            image((int)px, (int)py) = vec4(vec3(result), 1);
        } else {
            for (size_t i = 0; i < lanes; i++) {
                image((int)px[i], (int)py[i]) = vec4(vec3(result[i]), 1);
            }
        }
    };
    {
        Profiler profiler;
        tiled_for(500, 500, f);
        printf("%lfs\n", profiler.elapsed_seconds());
        default_image_writer()->write(image, "mandelbrot-vec.png", IdentityProcessor());
    }

    {
        Profiler profiler;
        scalar_for(500, 500, f);
        printf("%lfs\n", profiler.elapsed_seconds());
        default_image_writer()->write(image, "mandelbrot-scalar.png", IdentityProcessor());
    }
}
