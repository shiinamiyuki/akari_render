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

#include <Akari/Core/SIMD.hpp>
int main() {
    using namespace Akari;
    simd_array<float, 32> a, b;
    for (int i = 0; i < 32; i++) {
        a[i] = 2 * i + 1;
        b[i] = 3 * i + 2;
    }
    a = a + b;
    auto mask = array_operator_lt<float, 32>::apply(a, b);
    for (int i = 0; i < 32; i++) {
        printf("%f %f %d\n", a[i], b[i], mask[i]);
    }
    auto c = select(~(a<100.0f & a> 50.0f), a, b);
    for (int i = 0; i < 32; i++) {
        printf("%f %f %f %d\n", a[i], b[i], c[i], mask[i]);
    }

}
