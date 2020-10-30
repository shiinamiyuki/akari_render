

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
#include <akari/core/math.h>
#include <akari/render/scenegraph.h>
#include <akari/render/common.h>
#include <akari/core/memory.h>
#include <akari/core/variant.h>
#include <optional>

namespace akari::render {
    class PCGSampler;
    class LCGSampler;

    class PCGSampler {
        uint64_t state = 0x4d595df4d0f33173; // Or something seed-dependent
        static uint64_t const multiplier = 6364136223846793005u;
        static uint64_t const increment = 1442695040888963407u; // Or an arbitrary odd constant
        static AKR_XPU uint32_t rotr32(uint32_t x, unsigned r) { return x >> r | x << (-r & 31); }
        AKR_XPU uint32_t pcg32(void) {
            uint64_t x = state;
            unsigned count = (unsigned)(x >> 59); // 59 = 64 - 5

            state = x * multiplier + increment;
            x ^= x >> 18;                              // 18 = (64 - 27)/2
            return rotr32((uint32_t)(x >> 27), count); // 27 = 32 - 5
        }
        AKR_XPU void pcg32_init(uint64_t seed) {
            state = seed + increment;
            (void)pcg32();
        }

      public:
        AKR_XPU void set_sample_index(uint64_t idx) { pcg32_init(idx); }
        AKR_XPU Float next1d() { return Float(pcg32()) / (float)0xffffffff; }

        AKR_XPU void start_next_sample() {}
        AKR_XPU PCGSampler(uint64_t seed = 0u) { pcg32_init(seed); }
    };
    class LCGSampler {
        uint32_t seed;

      public:
        AKR_XPU void set_sample_index(uint64_t idx) { seed = idx & 0xffffffff; }
        AKR_XPU Float next1d() {
            seed = (1103515245 * seed + 12345);
            return (Float)seed / (Float)0xFFFFFFFF;
        }
        AKR_XPU void start_next_sample() {}
        AKR_XPU LCGSampler(uint64_t seed = 0u) : seed(seed) {}
    };
    class Sampler : public Variant<PCGSampler, LCGSampler> {
      public:
        using Variant::Variant;
        AKR_XPU Float inline next1d();
        AKR_XPU inline vec2 next2d();
        AKR_XPU inline void start_next_sample();
        AKR_XPU inline void set_sample_index(uint64_t idx);
    };
    class SamplerNode : public SceneGraphNode {
      public:
        virtual Sampler create_sampler() = 0;
    };

    class RandomSamplerNode : public SamplerNode {
        std::string generator = "pcg";

      public:
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {
            if (field == "generator") {
                AKR_ASSERT_THROW(value.is_string());
                auto s = value.get<std::string>().value();
                if (s == "pcg" || s == "pcg32") {
                    generator = "pcg";
                } else if (s == "lcg") {
                    generator = "lcg";
                } else {
                    throw std::runtime_error(fmt::format("unknown generator {}", s));
                }
            }
        }
        Sampler create_sampler() override {
            if (generator == "pcg") {
                return PCGSampler();
            } else if (generator == "lcg") {
                return LCGSampler();
            } else
                AKR_ASSERT_THROW(false);
        }
    };

    AKR_XPU inline Float Sampler::next1d() { AKR_VAR_DISPATCH(next1d); }
    AKR_XPU inline vec2 Sampler::next2d() { return vec2(next1d(), next1d()); }
    AKR_XPU inline void Sampler::start_next_sample() { AKR_VAR_DISPATCH(start_next_sample); }
    AKR_XPU inline void Sampler::set_sample_index(uint64_t idx) { AKR_VAR_DISPATCH(set_sample_index, idx); }
} // namespace akari::render