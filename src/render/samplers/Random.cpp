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

#include <akari/core/plugin.h>
#include <akari/render/sampler.h>
namespace akari {

    class RandomSampler : public Sampler {
        Rng rng;
        size_t dim = 0;

      public:
        RandomSampler() = default;
        Float next1d() override {
            dim++;
            return rng.uniformFloat();
        }
        std::shared_ptr<Sampler> clone() const override { return std::make_shared<RandomSampler>(*this); }
        void set_sample_index(size_t index) override { rng = Rng(index); }
        void start_next_sample() override { dim = 0; }
        size_t current_dimension() override { return dim; }
        AKR_DECL_COMP()
    };
    AKR_EXPORT_PLUGIN(RandomSampler, p){
        auto c = class_<RandomSampler, Sampler, Component>("RandomSampler");
        c.constructor<>();
        c.method("next1d", &RandomSampler::next1d);
        c.method("next2d", &RandomSampler::next2d);
        c.method("clone", &RandomSampler::clone);
    }
} // namespace akari