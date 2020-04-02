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

#ifndef AKARIRENDER_DISTRIBUTION_HPP
#define AKARIRENDER_DISTRIBUTION_HPP
#include <Akari/Core/Math.h>

namespace Akari {
    /*
     * Return the largest index i such that
     * pred(i) is true
     * If no such index i, last is returned
     * */
    template <typename Pred> int UpperBound(int first, int last, Pred &&pred) {
        int lo = first;
        int hi = last;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (pred(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return std::clamp<int>(hi-1,0,(last-first) - 2);
    }
    struct Distribution1D {
        Distribution1D(const Float *f, size_t n) : func(f, f + n), cdf(n + 1) {
            cdf[0] = 0;
            for (size_t i = 0; i < n; i++) {
                cdf[i + 1] = cdf[i] + func[i] / n;
            }
            funcInt = cdf[n];
            if (funcInt == 0) {
                for (uint32_t i = 1; i < n + 1; ++i)
                    cdf[i] = Float(i) / Float(n);
            } else {
                for (uint32_t i = 1; i < n + 1; ++i)
                    cdf[i] /= funcInt;
            }
        }
        // y = F^{-1}(u)
        // P(Y <= y) = P(F^{-1}(U) <= u) = P(U <= F(u)) = F(u)
        // Assume: 0 <= i < n
        [[nodiscard]] Float PdfDiscrete(int i) const { return func[i] / (funcInt * Count()); }
        [[nodiscard]] Float PdfContinuous(Float x) const {
            uint32_t offset = static_cast<uint32_t>(x * Count());
            return func[offset / funcInt];
        }
        int SampleDiscrete(Float u, Float *pdf = nullptr) {
            uint32_t i = UpperBound(0, cdf.size(), [=](int idx) { return cdf[idx] <= u; });
            if (pdf) {
                *pdf = PdfDiscrete(i);
            }
            return i;
        }

        Float SampleContinuous(Float u, Float *pdf = nullptr) {
            uint32_t offset = UpperBound(0, cdf.size(), [=](int idx) { return cdf[idx] <= u; });
            Float du = u - cdf[offset];
            if ((cdf[offset + 1] - cdf[offset]) > 0)
                du /= (cdf[offset + 1] - cdf[offset]);
            if (pdf)
                *pdf = func[offset] / funcInt;
            return ((float)offset + du) / Count();
        }

        [[nodiscard]] size_t Count() const { return func.size(); }
        [[nodiscard]] Float Integral()const{
            return funcInt;
        }
      private:
        std::vector<Float> func, cdf;
        Float funcInt;
    };
} // namespace Akari

#endif // AKARIRENDER_DISTRIBUTION_HPP
