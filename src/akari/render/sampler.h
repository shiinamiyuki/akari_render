

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
#include <optional>

namespace akari::render {
    struct Rng {
        Rng(uint64_t sequence = 0) { pcg32_init(sequence); }
        uint32_t uniform_u32() { return pcg32(); }
        double uniform_float() { return pcg32() / double(0xffffffff); }

      private:
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
    };
    class Sampler {
      public:
        virtual Float next1d() = 0;
        vec2 next2d() { return vec2(next1d(), next1d()); }
        virtual void start_next_sample() = 0;
        virtual void set_sample_index(uint64_t idx) = 0;
        virtual std::shared_ptr<Sampler> clone(Allocator<> allocator) const = 0;
        virtual ~Sampler() = default;
    };
    class SamplerNode : public SceneGraphNode {
      public:
        virtual std::shared_ptr<Sampler> create_sampler(Allocator<> allocator) = 0;
    };

    class PCGSampler : public Sampler {
        Rng rng;

      public:
        void set_sample_index(uint64_t idx) override { rng = Rng(idx); }
        Float next1d() override { return rng.uniform_float(); }

        void start_next_sample() override {}
        PCGSampler(uint64_t seed = 0u) : rng(seed) {}
        std::shared_ptr<Sampler> clone(Allocator<> allocator) const override {
            return make_pmr_shared<PCGSampler>(allocator, *this);
        }
    };
    class LCGSampler : public Sampler {
        uint32_t seed;

      public:
        void set_sample_index(uint64_t idx) override { seed = idx & 0xffffffff; }
        Float next1d() override {
            seed = (1103515245 * seed + 12345);
            return (Float)seed / (Float)0xFFFFFFFF;
        }
        void start_next_sample() override {}
        LCGSampler(uint64_t seed = 0u) : seed(seed) {}
        std::shared_ptr<Sampler> clone(Allocator<> allocator) const override {
            return make_pmr_shared<LCGSampler>(allocator, *this);
        }
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
        std::shared_ptr<Sampler> create_sampler(Allocator<> allocator) override {
            if (generator == "pcg") {
                return make_pmr_shared<PCGSampler>(allocator);
            } else if (generator == "lcg") {
                return make_pmr_shared<LCGSampler>(allocator);
            } else
                AKR_ASSERT_THROW(false);
        }
    };
} // namespace akari::render