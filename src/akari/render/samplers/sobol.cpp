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
#include <akari/render/sampler.h>

#define SOBOL_MAX_DIMENSION 21201
#define SOBOL_BITS 32
extern const unsigned int SobolMatrix[SOBOL_MAX_DIMENSION][SOBOL_BITS];

static inline uint32_t cmj_hash_simple(uint32_t i, uint32_t p) {
    i = (i ^ 61) ^ p;
    i += i << 3;
    i ^= i >> 4;
    i *= 0x27d4eb2d;
    return i;
}

#define SOBOL_SKIP 64
// https://wiki.blender.org/wiki/Source/Render/Cycles/Sobol
static inline float sobol(const unsigned int vectors[][32], unsigned int dimension, unsigned int i, unsigned int rng) {
    unsigned int result = 0;
    i += SOBOL_SKIP;
    for (unsigned int j = 0; i; i >>= 1, j++)
        if (i & 1)
            result ^= vectors[dimension][j];

    float r = result * (1.0f / (float)0xFFFFFFFF);
    uint32_t tmp_rng = cmj_hash_simple(dimension, rng);
    float shift = tmp_rng * (1.0f / (float)0xFFFFFFFF);
    return r + shift - floorf(r + shift);
}
namespace akari::render {
    class SobolSampler : public Sampler {
        int dimension = 0;
        int sample = 0;
        int rotation;

      public:
        void set_sample_index(uint64_t idx) override {
            Rng rng(idx);
            rotation = rng.uniform_u32();
            sample = -1;
        }
        Float next1d() override { return sobol(SobolMatrix, dimension++, sample, rotation); }
        void start_next_sample() override {
            dimension = 0;
            sample++;
        }
        SobolSampler() { rotation = 0; }
        std::shared_ptr<Sampler> clone(Allocator<> allocator) const override {
            return make_pmr_shared<SobolSampler>(allocator, *this);
        }
    };
    class SobolSamplerNode final : public SamplerNode {

      public:
        void object_field(sdl::Parser &parser, sdl::ParserContext &ctx, const std::string &field,
                          const sdl::Value &value) override {}
        std::shared_ptr<Sampler> create_sampler(Allocator<> allocator) override {
            return make_pmr_shared<SobolSampler>(allocator);
        }
    };
    AKR_EXPORT_NODE(SobolSampler, SobolSamplerNode)
} // namespace akari::render