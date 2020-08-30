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
#include <akari/common/taggedpointer.h>
#include <akari/common/smallarena.h>
namespace akari {
    AKR_VARIANT class RandomSampler {
        uint64_t state = 0x4d595df4d0f33173; // Or something seed-dependent
        static uint64_t const multiplier = 6364136223846793005u;
        static uint64_t const increment = 1442695040888963407u; // Or an arbitrary odd constant
        static uint32_t rotr32(uint32_t x, unsigned r) { return x >> r | x << (-r & 31); }
        uint32_t pcg32(void) {
            uint64_t x = state;
            unsigned count = (unsigned)(x >> 59); // 59 = 64 - 5

            state = x * multiplier + increment;
            x ^= x >> 18;                              // 18 = (64 - 27)/2
            return rotr32((uint32_t)(x >> 27), count); // 27 = 32 - 5
        }
        void pcg32_init(uint64_t seed) {
            state = seed + increment;
            (void)pcg32();
        }

      public:
        AKR_IMPORT_CORE_TYPES()
        void set_sample_index(uint64_t idx) { pcg32_init(idx); }
        Float next1d() { return Float(pcg32()) / (float)0xffffffff; }
        Point2f next2d() { return Point2f(next1d(), next1d()); }
        void start_next_sample() {}
        RandomSampler(uint64_t seed = 0u) { pcg32_init(seed); }
        RandomSampler *clone(SmallArena *arena) const { return arena->alloc<RandomSampler>(*this); }
    };
    AKR_VARIANT class Sampler : TaggedPointer<RandomSampler<Float, Spectrum>> {
      public:
        AKR_IMPORT_CORE_TYPES()
        using TaggedPointer<RandomSampler<Float, Spectrum>>::TaggedPointer;
        Float next1d() { AKR_TAGGED_DISPATCH(next1d); }
        Point2f next2d() { AKR_TAGGED_DISPATCH(next2d); }
        void start_next_sample() { AKR_TAGGED_DISPATCH(start_next_sample); }
        void set_sample_index(uint64_t idx) { AKR_TAGGED_DISPATCH(set_sample_index, idx); }
        Sampler clone(SmallArena *arena) const { AKR_TAGGED_DISPATCH(clone, arena); }
    };
} // namespace akari