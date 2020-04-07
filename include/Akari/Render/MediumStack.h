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

#ifndef AKARIRENDER_MEDIUMSTACK_H
#define AKARIRENDER_MEDIUMSTACK_H

#include <Akari/Core/Spectrum.h>

namespace Akari {
    class Medium;
    struct MediumRecord {
        Float eta = 1.0f;
        const Medium *medium = nullptr;
    };
    template <size_t N> struct TMediumStack {
        MediumRecord &operator[](size_t i) { return stack[i]; }
        const MediumRecord &operator[](size_t i) const { return stack[i]; }
        size_t size() const { return sp; }
        void push_back(const MediumRecord &record) { stack[sp++] = record; }
        void pop_back() { sp--; }
        MediumRecord &back() { return stack[sp - 1]; }
        const MediumRecord &back() const { return stack[sp - 1]; }

      private:
        std::array<MediumRecord, N> stack;
        size_t sp = 0;
    };

    using MediumStack = TMediumStack<8>;
} // namespace Akari
#endif // AKARIRENDER_MEDIUMSTACK_H
