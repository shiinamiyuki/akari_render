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

#ifndef AKARIRENDER_PROGRESS_HPP
#define AKARIRENDER_PROGRESS_HPP
#include <atomic>
#include <cstdio>
namespace Akari {
    inline void ShowProgress(double progress, size_t width) {
        printf("[");
        size_t pos = size_t(width * progress);
        for (size_t i = 0; i < width; ++i) {
            if (i < pos)
                printf("=");
            else if (i == pos)
                printf(">");
            else
                printf(" ");
        }
        printf("] %2d %%\r", int(progress * 100.0));
        fflush(stdin);
    }

    struct ProgressReporter {
        explicit ProgressReporter(size_t total)
            : total(total), callback([](size_t cur, size_t total) { ShowProgress(double(cur) / total, 70); }) {}
        ProgressReporter(size_t total, std::function<void(size_t, size_t)> &&cb) : total(total), callback(cb) {}
        void Update() {
            count++;
            callback(count, total);
        }

      private:
        std::atomic<size_t> count = 0;
        std::atomic<size_t> total;
        std::function<void(size_t, size_t)> callback;
    };
} // namespace Akari

#endif // AKARIRENDER_PROGRESS_HPP
